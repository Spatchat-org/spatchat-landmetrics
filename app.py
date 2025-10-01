# app.py

import os
import re
import json
import time
import threading
import random
import sys
import gradio as gr

import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pylandstats import Landscape
from dotenv import load_dotenv

# LLM providers
from huggingface_hub import InferenceClient
from together import Together
from together.error import RateLimitError, ServiceUnavailableError

# --- LLM: HF primary, Together fallback (router-only; no content surfaced) ---

load_dotenv()

def _choice_content(choice):
    """
    Extract assistant text from HF/Together pydantic/dict choices.
    Handles str or list-of-parts content.
    """
    msg = getattr(choice, "message", None)
    if msg is None and isinstance(choice, dict):
        msg = choice.get("message")

    content = None
    if msg is not None:
        if isinstance(msg, dict):
            content = msg.get("content")
        else:
            content = getattr(msg, "content", None)

    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        content = "".join(parts)

    return content or ""

def _delta_text(delta):
    if isinstance(delta, dict):
        return delta.get("content", "")
    return getattr(delta, "content", "")

HF_MODEL_DEFAULT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TOGETHER_MODEL_DEFAULT = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

class _SpacedCallLimiter:
    """Ensure at least `min_interval_seconds` between calls (process-wide)."""
    def __init__(self, min_interval_seconds: float):
        self.min_interval = float(min_interval_seconds)
        self._lock = threading.Lock()
        self._last = 0.0
    def wait(self):
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last = time.monotonic()

class UnifiedLLM:
    """
    Primary: Hugging Face (Serverless or Endpoint via HF_ENDPOINT_URL)
    Fallback: Together.ai (if TOGETHER_API_KEY set)
    Returns plain string content (JSON only; never surfaced directly to user).
    """
    def __init__(self):
        hf_model_or_url = (os.getenv("HF_ENDPOINT_URL") or HF_MODEL_DEFAULT).strip()
        hf_token = (os.getenv("HF_TOKEN") or "").strip()

        self.hf_client = InferenceClient(
            model=hf_model_or_url,
            token=hf_token,
            timeout=300,
        )

        self.together = None
        self.together_model = (os.getenv("TOGETHER_MODEL") or TOGETHER_MODEL_DEFAULT).strip()
        tg_key = (os.getenv("TOGETHER_API_KEY") or "").strip()
        if tg_key:
            self.together = Together(api_key=tg_key)
            self._tg_limiter = _SpacedCallLimiter(min_interval_seconds=100.0)  # ≈0.6 QPM

    def _hf_chat(self, messages, max_tokens=256, temperature=0.0, stream=False):
        tries, delay = 3, 2.2
        last_err = None
        for _ in range(tries):
            try:
                if hasattr(self.hf_client, "chat_completion"):
                    resp = self.hf_client.chat_completion(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=stream,
                    )
                    if stream:
                        text = "".join(_delta_text(ch.choices[0].delta) for ch in resp)
                    else:
                        text = _choice_content(resp.choices[0])
                    return text
                else:
                    # rare fallback: convert messages to prompt
                    prompt = self._messages_to_prompt(messages)
                    text = self.hf_client.text_generation(
                        prompt,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        stream=False,
                        return_full_text=False,
                    )
                    return text
            except Exception as e:
                last_err = e
                time.sleep(delay)
                delay *= 1.8
        raise last_err

    @staticmethod
    def _messages_to_prompt(messages):
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n")
            else:
                parts.append(f"<|assistant|>\n{content}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def chat(self, messages, temperature=0.0, max_tokens=256, stream=False):
        try:
            return self._hf_chat(messages, max_tokens=max_tokens, temperature=temperature, stream=stream)
        except Exception as hf_err:
            print(f"[LLM] HF primary failed: {hf_err}", file=sys.stderr)
            if self.together is None:
                raise

            # pace Together BEFORE first attempt
            self._tg_limiter.wait()
            backoff = 12.0
            for attempt in range(4):
                try:
                    resp = self.together.chat.completions.create(
                        model=self.together_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream,
                    )
                    return _choice_content(resp.choices[0])
                except (RateLimitError, ServiceUnavailableError):
                    if attempt == 3:
                        raise
                    time.sleep(backoff + random.uniform(0, 3))
                    backoff *= 1.8

llm = UnifiedLLM()

# --- Metric glossary (patch/class/landscape; acronyms only) ---
metric_definitions = {
    # Patch-level
    "area":  ("Patch Area (AREA)", "Area of each patch"),
    "perim": ("Patch Perimeter (PERIM)", "Perimeter of each patch"),
    "para":  ("Perimeter–Area Ratio (PARA)", "p/a per patch"),
    "shape": ("Shape Index (SHAPE)", "Normalized shape per patch"),
    "frac":  ("Fractal Dimension (FRAC)", "Fractal complexity per patch"),
    "enn":   ("Euclidean Nearest Neighbor (ENN)", "Nearest same-class patch"),
    "core_p":("Core Area (CORE)", "Core area per patch"),
    "ncore": ("Number of Core Areas (NCORE)", "Disjunct cores per patch"),
    "cai":   ("Core Area Index (CAI)", "CORE / AREA per patch"),

    # Class & Landscape (area/edge)
    "ta":    ("Total Area (TA)", "Total area (class/landscape)"),
    "pland": ("Proportion of Landscape (PLAND)", "% of landscape by class"),
    "np":    ("Number of Patches (NP)", "Total patches"),
    "pd":    ("Patch Density (PD)", "Patches per area"),
    "lpi":   ("Largest Patch Index (LPI)", "% landscape by largest patch"),
    "te":    ("Total Edge (TE)", "Total edge length"),
    "edge_density": ("Edge Density (ED)", "Edge length per area unit"),

    # Core family (class & landscape unless noted)
    "tca":   ("Total Core Area (TCA)", "Sum of core areas"),
    "cpland":("Core Area % of Landscape (CPLAND)", "% core area of landscape (per class)"),
    "ndca":  ("Number of Disjunct Core Areas (NDCA)", "Count of disjunct cores"),
    "dcad":  ("Disjunct Core Area Density (DCAD)", "Core areas per area unit"),

    # Aggregation/connectivity (class & landscape)
    "lsi":   ("Landscape Shape Index (LSI)", "Standardized edginess"),
    "mesh":  ("Effective Mesh Size (MESH)", "Aggregation based on patch sizes"),

    # Diversity / contagion (landscape-only)
    "entropy":                    ("Entropy (H)", "Information-theoretic diversity"),
    "joint_entropy":              ("Joint Entropy H(X,Y)", "Adjacency complexity"),
    "conditional_entropy":        ("Conditional Entropy H(Y|X)", "Spatial complexity"),
    "mutual_information":         ("Mutual Information (MI)", "Aggregation"),
    "relative_mutual_information":("Relative Mutual Information (RMI)", "Normalized MI"),
    "shdi":                       ("Shannon Diversity Index (SHDI)", "Diversity"),
    "contag":                     ("Contagion (CONTAG)", "Aggregation of adjacencies"),
}

# --- Synonyms (user text → canonical codes) ---
synonyms = {
    # Patch-level
    "area":  ["area", "patch area", "area_patch", "area (patch)"],
    "perim": ["perimeter", "perim", "patch perimeter"],
    "para":  ["para", "perimeter-area ratio", "perimeter area ratio"],
    "shape": ["shape", "shape index"],
    "frac":  ["frac", "fractal dimension"],
    "enn":   ["enn", "nearest neighbor", "nearest-neighbor"],
    "core_p":["core", "core area", "core (patch)", "patch core"],
    "ncore": ["ncore", "number of core areas", "num core areas"],
    "cai":   ["cai", "core area index"],

    # Class & landscape
    "ta":    ["ta", "total area"],
    "pland": ["pland", "proportion of landscape", "percent landscape", "% landscape",
              "percent of landscape", "percent cover", "percentage landscape", "percent composition"],
    "np":    ["np", "number of patches"],
    "pd":    ["pd", "patch density"],
    "lpi":   ["lpi", "largest patch index"],
    "te":    ["te", "total edge"],
    "edge_density": ["ed", "edge density"],

    # Core family (class/landscape)
    "tca":   ["tca", "total core area"],
    "cpland":["cpland", "core area proportion of landscape", "core area percent of landscape"],
    "ndca":  ["ndca", "number of disjunct core areas"],
    "dcad":  ["dcad", "disjunct core area density"],

    # Aggregation/connectivity
    "lsi":   ["lsi", "landscape shape index"],
    "mesh":  ["mesh", "effective mesh size"],

    # Diversity/contagion (landscape-only)
    "entropy":                    ["entropy", "h"],
    "joint_entropy":              ["joint entropy", "h(x,y)", "hxy", "joinent"],
    "conditional_entropy":        ["conditional entropy", "h(y|x)", "hyx", "condent"],
    "mutual_information":         ["mutual information", "mi", "mutinf"],
    "relative_mutual_information":["relative mutual information", "rmi", "relmutinf"],
    "shdi":                       ["shdi", "shannon diversity"],
    "contag":                     ["contag", "contagion"],
}
reverse_synonyms = {}
for code, syn_list in synonyms.items():
    for syn in syn_list:
        reverse_synonyms[syn.lower()] = code

# --- Column/method mappings ---
# Class/Landscape metrics (strings are PyLandStats method names)
metric_map = {
    "ta":           "total_area",
    "pland":        "proportion_of_landscape",   # class only
    "np":           "number_of_patches",         # cross
    "pd":           "patch_density",             # cross
    "lpi":          "largest_patch_index",       # cross
    "te":           "total_edge",                # cross
    "edge_density": "edge_density",              # cross
    "tca":          "total_core_area",           # cross
    "cpland":       "core_area_proportion_of_landscape",  # class only (derive landscape)
    "ndca":         "number_of_disjunct_core_areas",      # cross
    "dcad":         "disjunct_core_area_density",         # cross
    "lsi":          "landscape_shape_index",     # cross
    "mesh":         "effective_mesh_size",       # cross

    # Landscape-only
    "entropy":                    "entropy",
    "joint_entropy":              "joint_entropy",
    "conditional_entropy":        "conditional_entropy",
    "mutual_information":         "mutual_information",
    "relative_mutual_information":"relative_mutual_information",
    "shdi":                       "shannon_diversity_index",
    "contag":                     "contagion",
}

# Patch-level metrics are computed via compute_patch_metrics_df
patch_metric_map = {
    "area":  "area",
    "perim": "perimeter",
    "para":  "perimeter_area_ratio",
    "shape": "shape_index",
    "frac":  "fractal_dimension",
    "enn":   "euclidean_nearest_neighbor",
    "core_p":"core_area",
    "ncore": "number_of_core_areas",
    "cai":   "core_area_index",
}

# Helper (callable as attribute on Landscape for scalars)
helper_methods = metric_map.copy()

# Level partitions
patch_only = set(patch_metric_map.keys())
class_only = {"pland", "cpland"}           # strictly class by API (we derive CPLAND at landscape)
cross_level = {"ta","np","pd","lpi","te","edge_density","tca","ndca","dcad","lsi","mesh"}
landscape_only = {"entropy","joint_entropy","conditional_entropy",
                  "mutual_information","relative_mutual_information","shdi","contag"}

# --- Raster preview helpers (used by UI) ---
def no_raster_fig():
    fig, ax = plt.subplots(figsize=(5,5))
    ax.text(0.5,0.5,"🗂️ No raster loaded.",
            ha='center',va='center',color='gray',fontsize=14)
    ax.set_title("Raster Preview", color='dimgray')
    ax.axis('off')
    return fig

def preview_raster(file):
    if not file: return no_raster_fig()
    with rasterio.open(file.name) as src:
        raw = src.read(1); nodata = src.nodata or 0
    if raw.dtype.kind in ('U','S','O'):
        data = raw.astype(str)
        uniq = np.unique(data[data!=""])
        arr  = np.zeros_like(raw,dtype=int)
        for i,nm in enumerate(uniq,1):
            arr[data==nm] = i
        labels = [f"{i}: {nm}" for i,nm in enumerate(uniq,1)]
    else:
        arr = raw
        vals = np.unique(arr[arr!=nodata])
        labels = [f"Class {int(v)}" for v in vals]
    n = len(labels)
    colors = plt.cm.tab10(np.linspace(0,1,n))
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(arr,cmap='tab10'); ax.set_title("Uploaded Raster"); ax.axis('off')
    handles = [mpatches.Patch(color=c,label=l) for c,l in zip(colors,labels)]
    ax.legend(handles=handles,loc='lower left',fontsize='small')
    return fig

def clear_raster():
    return None, gr.update(value=no_raster_fig(), visible=True)

def notify_upload(file, history):
    if file:
        return history + [{"role":"assistant","content":"📥 Raster uploaded successfully!"}], ""
    return history, ""

# --- Core metric / metadata helpers (pylandstats-backed only) ---
def answer_metadata(file):
    with rasterio.open(file.name) as src:
        return (
            f"CRS: {src.crs}\n"
            f"Resolution: {src.res[0]:.2f}×{src.res[1]:.2f}\n"
            f"Extent: {src.bounds}\n"
            f"Bands: {src.count}\n"
            f"NoData: {src.nodata}"
        )

def count_classes(file):
    with rasterio.open(file.name) as src:
        arr = src.read(1); nodata = src.nodata or 0
    vals = np.unique(arr[arr!=nodata])
    return f"Your raster contains {len(vals)} unique classes."

def list_metrics_text():
    lines = []
    lines.append("**Patch-level:**")
    lines += [f"- {metric_definitions[k][0]} (`{k}`)" for k in sorted(patch_only)]

    lines.append("\n**Class-level:**")
    lines += [f"- {metric_definitions[k][0]} (`{k}`)" for k in sorted(class_only)]

    lines.append("\n**Cross-level (class & landscape):**")
    lines += [f"- {metric_definitions[k][0]} (`{k}`)" for k in sorted(cross_level)]

    lines.append("\n**Landscape-only:**")
    lines += [f"- {metric_definitions[k][0]} (`{k}`)" for k in sorted(landscape_only)]
    return "\n".join(lines)

def _build_landscape(file):
    with rasterio.open(file.name) as src:
        raw = src.read(1)
        x_res,y_res = src.res
        nodata = src.nodata or 0
    if raw.dtype.kind in ("U","S","O"):
        data_str   = raw.astype(str)
        uniq_names = np.unique(data_str[data_str!=""])
        name2code  = {nm:i+1 for i,nm in enumerate(uniq_names)}
        arr        = np.zeros_like(raw,dtype=int)
        for nm,code in name2code.items():
            arr[data_str==nm] = code
        return Landscape(arr, res=(x_res,y_res), nodata=0)
    else:
        return Landscape(file.name, nodata=nodata, res=(x_res,y_res))

def compute_landscape_only_text(file, keys):
    ls = _build_landscape(file)
    parts = ["**Landscape-level metrics:**"]
    for key in keys:
        name,_ = metric_definitions[key]
        if key == "np":
            df = ls.compute_class_metrics_df(metrics=["number_of_patches"])
            val = int(df["number_of_patches"].sum())
        elif key == "cpland":
            # Derived: sum of class TCA divided by TA, in %
            df = ls.compute_class_metrics_df(metrics=["total_core_area","total_area"])
            tca_sum = float(df["total_core_area"].sum())
            ta_sum  = float(df["total_area"].sum())
            val = 0.0 if ta_sum == 0 else (100.0 * tca_sum / ta_sum)
        elif key in helper_methods:
            val = getattr(ls, helper_methods[key])()
        else:
            df  = ls.compute_landscape_metrics_df(metrics=[metric_map[key]])
            val = df[metric_map[key]].iloc[0]
        parts.append(
            f"**{name} ({key.upper()}):** {val:.4f}"
            if isinstance(val, float) else f"**{name} ({key.upper()}):** {val}"
        )
    return "\n\n".join(parts)

def compute_class_only_text(file, keys):
    ls   = _build_landscape(file)
    cols = [metric_map[k] for k in keys]
    df   = ls.compute_class_metrics_df(metrics=cols)
    # Ensure class identifier column exists for display
    if "class_val" not in df.columns:
        df = df.rename_axis("class_val").reset_index()
    else:
        df = df.reset_index(drop=True)
    df["class_name"] = df["class_val"].astype(int).apply(lambda c: f"Class {c}")
    display_cols = ["class_name"] + [metric_map[k] for k in keys]
    tbl  = df[display_cols].to_markdown(index=False)
    return f"**Class-level metrics:**\n{tbl}"

def compute_patch_only_text(file, keys, row_cap=50):
    ls   = _build_landscape(file)
    cols = [patch_metric_map[k] for k in keys]
    df   = ls.compute_patch_metrics_df(metrics=cols).reset_index()
    total_rows = len(df)
    if total_rows > row_cap:
        df = df.head(row_cap)
        note = f"\n\n_Showing first {row_cap} of {total_rows} patches._"
    else:
        note = ""
    tbl  = df.to_markdown(index=False)
    return f"**Patch-level metrics:**\n{tbl}{note}"

def compute_multiple_metrics_text(file, keys):
    land_keys  = [k for k in keys if (k in cross_level or k in landscape_only)]
    class_keys = [k for k in keys if k in class_only]
    patch_keys = [k for k in keys if k in patch_only]
    parts = []
    if land_keys:
        parts.append(compute_landscape_only_text(file, land_keys))
    if class_keys:
        parts.append(compute_class_only_text(file, class_keys))
    if patch_keys:
        parts.append(compute_patch_only_text(file, patch_keys))
    return "\n\n".join(parts)

# --- Tool wrappers (UI expects these) ---
def run_metadata(file):
    return answer_metadata(file)

def run_count_classes(file):
    return count_classes(file)

def run_list_metrics(file):
    return list_metrics_text()

def run_compute_metrics(file, raw_metrics, level):
    # 1) normalize & map synonyms, catch unknowns
    mapped, unknown = [], []
    for m in raw_metrics:
        ml = str(m).lower()
        if ml in metric_definitions:
            cand = ml
        elif ml in reverse_synonyms:
            cand = reverse_synonyms[ml]
        else:
            unknown.append(m)
            continue

        # If user explicitly asked for patch level and said "core", use patch variant
        if level == "patch" and cand == "core":
            cand = "core_p"

        if (cand in metric_map) or (cand in patch_metric_map):
            mapped.append(cand)
        else:
            unknown.append(m)

    if unknown:
        return f"Sorry, I don’t recognize: {', '.join(map(str, unknown))}. Could you clarify?"

    if level == "patch":
        keys = [k for k in mapped if k in patch_only]
        return compute_patch_only_text(file, keys) if keys else "No patch-level metrics requested."

    # split into cross-level vs class-only
    land = [c for c in mapped if (c in cross_level or c in landscape_only)]
    clas = [c for c in mapped if c in class_only]
    any_cross = bool(set(mapped) & cross_level)

    # explicit class request
    if level == "class":
        return compute_class_only_text(file, clas or [m for m in mapped if m in (class_only | cross_level)])

    # explicit landscape request
    if level == "landscape":
        if not land and clas:
            return compute_class_only_text(file, clas)
        return compute_landscape_only_text(file, land or [m for m in mapped if m in (cross_level | landscape_only)])

    # infer / both
    if any_cross:
        return compute_multiple_metrics_text(file, mapped)
    # else class-only
    return compute_class_only_text(file, clas or mapped)

# --- Strict, JSON-only system prompt (router) ---
SYSTEM_PROMPT = """
You are a tool router. Output ONLY a single JSON object with this exact schema:
{"tool":"metadata"|"count_classes"|"list_metrics"|"compute_metrics",
 "level":"landscape"|"class"|"patch"|"both",
 "metrics":[<lowercase metric codes>]}
Rules:
- NEVER include any other text, Markdown, or numbers.
- NEVER estimate or compute values.
- If unsure, choose {"tool":"compute_metrics","level":"both","metrics":[]}.
""".strip()

# --- Deterministic intent parser (covers PLAND/ED, etc.) ---
LEVEL_HINTS = {
    "class": re.compile(r"\bclass\b|\bby class\b|\bclass[-\s]*level\b"),
    "landscape": re.compile(r"\blandscape\b|\boverall\b|\bwhole map\b|\blandscape[-\s]*level\b"),
    "patch": re.compile(r"\bpatch\b|\bpatch[-\s]*level\b")
}

def parse_request_rule_based(user_msg: str):
    """
    Return (level, metrics) if we can confidently parse; else (None, []).
    level in {"class","landscape","patch","both"}
    """
    lower = user_msg.lower()

    # Level
    want_class = bool(LEVEL_HINTS["class"].search(lower))
    want_land  = bool(LEVEL_HINTS["landscape"].search(lower))
    want_patch = bool(LEVEL_HINTS["patch"].search(lower))

    if sum([want_class, want_land, want_patch]) > 1:
        level = "both"
    elif want_patch:
        level = "patch"
    elif want_class:
        level = "class"
    elif want_land:
        level = "landscape"
    else:
        level = "both"  # default

    found = set()
    text = f" {lower} "

    # 1) phrase matches (synonyms can contain spaces)
    for canon, syns in synonyms.items():
        for s in sorted(syns, key=len, reverse=True):
            if f" {s} " in text:
                found.add(canon); break

    # 2) explicit metric codes inside parentheses, e.g. "(pland)"
    for par in re.findall(r"\(([a-zA-Z_]+)\)", lower):
        if par in metric_definitions:
            found.add(par)
        elif par in reverse_synonyms:
            found.add(reverse_synonyms[par])

    # 3) single-token canonical/synonym names (allow (), |)
    for tok in re.findall(r"[a-zA-Z_%()|]+", lower):
        if tok in metric_definitions:
            found.add(tok)
        elif tok in reverse_synonyms:
            found.add(reverse_synonyms[tok])

    # resolve 'core' depending on requested level
    mapped = []
    for m in found:
        if m == "core_p" or (m == "core" and level == "patch"):
            mapped.append("core_p")
        elif m == "core":
            mapped.append("tca")
        else:
            mapped.append(m)

    # Keep only implementable codes
    mapped = [m for m in mapped if (m in metric_map or m in patch_metric_map)]
    return (level if mapped else None, mapped)

# --- extra guard against suspicious tables in replies ---
def _looks_like_hallucinated_table(txt: str) -> bool:
    suspicious = ["area_km2", "area percent", "percent landscape"]  # computed only under PLAND/class flow
    # We allow "Proportion of Landscape (PLAND)" header in our own text; this just catches ad-hoc tables
    return any(s in txt.lower() for s in suspicious) and ("| class" in txt.lower() or "\t" in txt)

# --- Welcome message shown on load ---
WELCOME_MESSAGE = """\
👋 Hi! I’m Spatchat. I am an expert in quantifying landscape metrics.

This is a list of metrics I can calculate:

Patch level
- Area/Edge: AREA, PERIM
- Shape: PARA, SHAPE, FRAC
- Core: CORE, NCORE, CAI
- Aggregation: ENN

Class level
- Area/Edge: TA, PLAND, NP, PD, LPI, TE, ED
- Core: TCA, CPLAND, NDCA, DCAD
- Aggregation/Connectivity: LSI, MESH

Landscape level
- Area/Edge summaries: TA, NP, PD, LPI, TE, ED, LSI
- Core summaries: TCA, CPLAND, NDCA, DCAD
- Richness/Diversity: H (entropy), SHDI
- Contagion/Interspersion: H(X,Y) (joint entropy), H(Y|X) (conditional entropy), MI (mutual information), RMI (relative MI), CONTAG

You can ask me in natural language. For example:
- "What is the percentage of landscape (PLAND) for class 1 at the class level?"
- "Give me PLAND for all classes at the class level."
- "What is Shannon’s diversity index (SHDI) at the landscape level?"
- "Tell me the largest patch index (LPI) at the landscape level."
- "Compute patch area (AREA) at the patch level."

If I’m not sure what you mean, I’ll suggest close matches and ask you to confirm.
Upload a GeoTIFF to begin — then ask me for any landscape metric.
"""

# --- Main handler used by UI ---

def analyze_raster(file, user_msg, history):
    clear = ""  # clear the textbox after submit
    hist = history + [{"role":"user","content":user_msg}]
    lower = user_msg.lower()

    if file is None:
        hist.append({"role":"assistant","content":
                     "Please upload a GeoTIFF before asking anything."})
        return hist, clear

    # shortcuts
    if re.search(r"\ball landscape(?: level)? metrics\b", lower):
        keys = sorted((cross_level | landscape_only))
        reply = compute_landscape_only_text(file, keys)
        hist.append({"role":"assistant","content":reply})
        return hist, clear

    if re.search(r"\ball class(?: level)? metrics\b", lower):
        keys = sorted(class_only | cross_level)
        reply = compute_class_only_text(file, keys)
        hist.append({"role":"assistant","content":reply})
        return hist, clear

    if re.search(r"\ball patch(?: level)? metrics\b", lower):
        keys = sorted(patch_only)
        reply = compute_patch_only_text(file, keys)
        hist.append({"role":"assistant","content":reply})
        return hist, clear

    if re.search(r"\ball metrics\b", lower):
        keys = sorted(set(metric_map.keys()) | set(patch_metric_map.keys()))
        reply = compute_multiple_metrics_text(file, keys)
        hist.append({"role":"assistant","content":reply})
        return hist, clear

    # 1) Deterministic parser (no LLM)
    lvl_rb, mets_rb = parse_request_rule_based(user_msg)
    if lvl_rb is not None and mets_rb:
        reply = run_compute_metrics(file, mets_rb, lvl_rb)
        if _looks_like_hallucinated_table(reply):
            reply = "Let’s specify the metric explicitly (e.g., `class: pland`)."
        hist.append({"role":"assistant","content":reply})
        return hist, clear

    # 2) LLM router
    resp = llm.chat(
        messages=[{"role":"system","content":SYSTEM_PROMPT}] + hist,
        temperature=0.0,
        max_tokens=128,
        stream=False
    )

    try:
        call = json.loads(resp)
        if not isinstance(call, dict):
            raise ValueError("non-dict")
        tool = call.get("tool")
        if tool not in {"metadata","count_classes","list_metrics","compute_metrics"}:
            raise ValueError("bad tool")
    except Exception:
        hist.append({"role":"assistant","content":
                     "Which metric and level would you like? For example: `patch: area`, `class: pland`, or `landscape: edge density`."})
        return hist, clear

    if tool == "metadata":
        reply = run_metadata(file)
    elif tool == "count_classes":
        reply = run_count_classes(file)
    elif tool == "list_metrics":
        reply = run_list_metrics(file)
    elif tool == "compute_metrics":
        raw_metrics = call.get("metrics", [])
        level       = call.get("level", "both")
        mapped, unknowns = [], []
        for m in (raw_metrics or []):
            ml = str(m).lower()
            if ml in metric_definitions:
                mapped.append(ml)
            elif ml in reverse_synonyms:
                mapped.append(reverse_synonyms[ml])
            else:
                unknowns.append(m)
        if unknowns or not mapped:
            reply = "I didn’t recognize those metrics. Try one of: area (patch), pland (class), edge density (landscape)."
        else:
            reply = run_compute_metrics(file, mapped, level)
    else:
        reply = "I didn’t catch which metric you want. Try: `class: pland`."

    if _looks_like_hallucinated_table(reply):
        reply = "Let’s specify the metric explicitly (e.g., `class: pland`)."

    hist.append({"role":"assistant","content":reply})
    return hist, clear

# --- UI setup & launch ---

initial_history = [{"role": "assistant", "content": WELCOME_MESSAGE}]

with gr.Blocks(title="Spatchat") as iface:
    gr.HTML('<head><link rel="icon" href="logo1.png"></head>')
    gr.Image(value="logo_long1.png", type="filepath",
             show_label=False, show_download_button=False,
             show_share_button=False, elem_id="logo-img")
    gr.HTML("<style>#logo-img img{height:90px;margin:10px;border-radius:6px;}</style>")
    gr.Markdown("## 🌲 Spatchat: Landscape Metrics Assistant {landmetrics}")
    gr.HTML("""
      <div style="margin-top: -10px; margin-bottom: 15px;">
        <input type="text" value="https://spatchat.org/browse/?room=landmetrics" id="shareLink" readonly style="width: 50%; padding: 5px; background-color: #f8f8f8; color: #222; font-weight: 500; border: 1px solid #ccc; border-radius: 4px;">
        <button onclick="navigator.clipboard.writeText(document.getElementById('shareLink').value)" style="padding: 5px 10px; background-color: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer;">
          📋 Copy Share Link
        </button>
        <div style="margin-top: 10px; font-size: 14px;">
          <b>Share:</b>
          <a href="https://twitter.com/intent/tweet?text=Checkout+Spatchat!&url=https://spatchat.org/browse/?room=landmetrics" target="_blank">🐦 Twitter</a> |
          <a href="https://www.facebook.com/sharer/sharer.php?u=https://spatchat.org/browse/?room=landmetrics" target="_blank">📘 Facebook</a>
        </div>
      </div>
    """)
    gr.HTML("""
      <div style="font-size: 14px;">
      © 2025 Ho Yi Wan & Logan Hysen. All rights reserved.<br>
      If you use Spatchat in research, please cite:<br>
      <b>Wan, H.Y.</b> & <b>Hysen, L.</b> (2025). <i>Spatchat: Landscape Metrics.</i>
      </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload GeoTIFF", type="filepath",
                                  file_types=[".tif", ".tiff"])
            raster_out = gr.Plot(label="Raster Preview")
            clear_btn = gr.Button("Clear Raster")
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(value=initial_history, type="messages",
                                 label="Spatchat Dialog")
            txt_in = gr.Textbox(placeholder="e.g. calculate edge density",
                                lines=1)

    # Enable queue (older-Gradio-safe signature)
    iface.queue(max_size=16)

    file_input.change(preview_raster, inputs=file_input, outputs=raster_out)
    file_input.change(notify_upload, inputs=[file_input, chatbot],
                      outputs=[chatbot, txt_in])
    clear_btn.click(clear_raster, inputs=None,
                    outputs=[file_input, raster_out])
    txt_in.submit(analyze_raster,
                  inputs=[file_input, txt_in, chatbot],
                  outputs=[chatbot, txt_in])

iface.launch(ssr_mode=False)