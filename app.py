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

# --- LLM: HF primary, Together fallback (with pacing & robust parsing) ---

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
    Returns plain string content.
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

# --- Metric glossary ---
metric_definitions = {
    "pland":        ("Proportion of Landscape (PLAND)", "% of landscape by class"),
    "np":           ("Number of Patches (NP)", "Total patches"),
    "pd":           ("Patch Density (PD)", "Patches per 100 ha"),
    "lpi":          ("Largest Patch Index (LPI)", "% landscape by largest patch"),
    "te":           ("Total Edge (TE)", "Total edge length"),
    "edge_density": ("Edge Density (ED)", "Edge length per hectare"),
    "lsi":          ("Landscape Shape Index (LSI)", "Shape complexity"),
    "tca":          ("Total Core Area (TCA)", "Sum core areas"),
    "mesh":         ("Effective Mesh Size (MESH)", "Average patch size"),
    "contag":       ("Contagion Index (CONTAG)", "Clumpiness"),
    "shdi":         ("Shannon Diversity Index (SHDI)", "Diversity"),
    "shei":         ("Shannon Evenness Index (SHEI)", "Evenness"),
    "area":         ("Total Area (AREA)", "Patch area"),
    "perim":        ("Total Perimeter (PERIM)", "Patch perimeter"),
    "para":         ("Perimeter-Area Ratio (PARA)", "Perimeter/area"),
    "shape":        ("Shape Index (SHAPE)", "Normalized shape"),
    "frac":         ("Fractal Dimension (FRAC)", "Fractal complexity"),
    "enn":          ("Euclidean Nearest Neighbor (ENN)", "Nearest patch distance"),
    "core":         ("Total Core Area (CORE)", "Interior area"),
    "nca":          ("Number of Core Areas (NCA)", "Count cores"),
    "cai":          ("Core Area Index (CAI)", "% core to area"),
}

# --- Synonyms ---
synonyms = {
    "edge_density": ["ed", "edge density"],
    "pland":        ["pland", "proportion of landscape"],
    "np":           ["np", "number of patches"],
    "pd":           ["pd", "patch density"],
    "lpi":          ["lpi", "largest patch index"],
    "te":           ["te", "total edge"],
}
reverse_synonyms = {}
for code, syn_list in synonyms.items():
    for syn in syn_list:
        reverse_synonyms[syn.lower()] = code

# --- Column mapping & helpers ---
metric_map = {
    "pland":        "proportion_of_landscape",
    "np":           "number_of_patches",
    "pd":           "patch_density",
    "lpi":          "largest_patch_index",
    "te":           "total_edge",
    "edge_density":"edge_density",
    "lsi":          "landscape_shape_index",
    "tca":          "total_core_area",
    "mesh":         "effective_mesh_size",
    "contag":       "contagion",
    "shdi":         "shannon_diversity_index",
    "shei":         None,
    "area":         "total_area",
    "perim":        "perimeter",
    "para":         "perimeter_area_ratio",
    "shape":        "shape_index",
    "frac":         "fractal_dimension",
    "enn":          "euclidean_nearest_neighbor",
    "core":         "total_core_area",
    "nca":          "number_of_core_areas",
    "cai":          "core_area_index",
}
helper_methods = {k:v for k,v in metric_map.items() if v}
col_map = helper_methods.copy()
class_only = {"pland","area","perim","para","shape","frac","enn","core","nca","cai"}
cross_level = ["np","pd","lpi","te","edge_density"]
landscape_only = [
    k for k in helper_methods
    if k not in class_only and k not in cross_level
]

# --- Raster preview helpers ---
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

# --- Core metric / metadata helpers ---
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
    lines = ["**Cross-level metrics:**"]
    lines += [f"- {metric_definitions[k][0]} (`{k}`)" for k in cross_level]
    lines.append("\n**Landscape-only:**")
    lines += [f"- {metric_definitions[k][0]} (`{k}`)" for k in helper_methods]
    lines.append("\n**Class-only:**")
    lines += [f"- {metric_definitions[k][0]} (`{k}`)" for k in class_only]
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
        if key=="np":
            df = ls.compute_class_metrics_df(metrics=["number_of_patches"])
            val = int(df["number_of_patches"].sum())
        elif key in helper_methods:
            val = getattr(ls, helper_methods[key])()
        else:
            df  = ls.compute_landscape_metrics_df(metrics=[col_map[key]])
            val = df[col_map[key]].iloc[0]
        parts.append(
            f"**{name} ({key.upper()}):** {val:.4f}"
            if isinstance(val, float)
            else f"**{name} ({key.upper()}):** {val}"
        )
    return "\n\n".join(parts)

def compute_class_only_text(file, keys):
    ls   = _build_landscape(file)
    cols = [col_map[k] for k in keys]
    df   = ls.compute_class_metrics_df(metrics=cols).rename_axis("code").reset_index()
    df["class_name"] = df["code"].astype(int).apply(lambda c: f"Class {c}")
    tbl  = df[["class_name"] + cols].to_markdown(index=False)
    return f"**Class-level metrics:**\n{tbl}"

def compute_multiple_metrics_text(file, keys):
    land_keys  = [k for k in keys if k in cross_level or k in landscape_only]
    class_keys = [k for k in keys if k in class_only]
    parts = []
    if land_keys:
        parts.append(compute_landscape_only_text(file, land_keys))
    if class_keys:
        parts.append(compute_class_only_text(file, class_keys))
    return "\n\n".join(parts)

# --- Tool wrappers ---
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
        ml = m.lower()
        if ml in metric_definitions:
            cand = ml
        elif ml in reverse_synonyms:
            cand = reverse_synonyms[ml]
        else:
            unknown.append(m)
            continue
        if metric_map.get(cand) is None:
            unknown.append(m)
        else:
            mapped.append(cand)
    if unknown:
        return f"Sorry, I don’t recognize: {', '.join(unknown)}. Could you clarify?"

    # split into cross-level vs class-only
    land = [c for c in mapped if c not in class_only]
    clas = [c for c in mapped if c in class_only]
    any_cross = bool(set(mapped) & set(cross_level))

    # explicit class request
    if level == "class":
        return compute_class_only_text(file, mapped)
    # explicit landscape request
    if level == "landscape":
        if not land and clas:
            return compute_class_only_text(file, clas)
        return compute_landscape_only_text(file, land)
    # infer: cross-level -> both
    if any_cross:
        return compute_multiple_metrics_text(file, mapped)
    # else class-only
    return compute_class_only_text(file, mapped)

# --- System & Fallback Prompts ---
SYSTEM_PROMPT = """
You are Spatchat, a friendly GIS assistant and expert in landscape metrics using PyLandStats.
Whenever the user asks for metadata (crs, resolution, extent, bands, nodata), reply _only_ with:
{"tool":"metadata"}

Whenever they ask “how many classes” or similar, reply:
{"tool":"count_classes"}

If they ask “list metrics” or “available metrics”, reply:
{"tool":"list_metrics"}

If they ask to compute metrics, reply exactly:
{"tool":"compute_metrics","level":"landscape"|"class"|"both","metrics":[<codes>]}
Do NOT invent numbers—your Python functions will compute them.
If you see unknown codes, ask the user to clarify.
""".strip()

FALLBACK_PROMPT = """
You are Spatchat, a landscape-metrics expert.
Keep replies under two sentences.
If you can’t map a request to one of your tools, ask the user to clarify.
""".strip()

# --- Main handler ---
def analyze_raster(file, user_msg, history):
    hist = history + [{"role":"user","content":user_msg}]
    lower = user_msg.lower()

    if file is None:
        hist.append({"role":"assistant","content":
                     "Please upload a GeoTIFF before asking anything."})
        return hist, ""

    # shortcuts
    if re.search(r"\ball landscape(?: level)? metrics\b", lower):
        keys = [k for k in metric_map if metric_map.get(k) and k not in class_only]
        return hist + [{"role":"assistant","content":compute_landscape_only_text(file, keys)}], ""
    if re.search(r"\ball class(?: level)? metrics\b", lower):
        keys = list(class_only)
        return hist + [{"role":"assistant","content":compute_class_only_text(file, keys)}], ""
    if re.search(r"\ball metrics\b", lower):
        keys = [k for k in metric_map if metric_map.get(k)]
        return hist + [{"role":"assistant","content":compute_multiple_metrics_text(file, keys)}], ""

    # LLM tool chooser (HF primary, Together fallback)
    resp = llm.chat(
        messages=[{"role":"system","content":SYSTEM_PROMPT}] + hist,
        temperature=0.0,
        max_tokens=256,
        stream=False
    )

    try:
        call = json.loads(resp)
    except Exception:
        conv = llm.chat(
            messages=[{"role":"system","content":FALLBACK_PROMPT}] + hist,
            temperature=0.7,
            max_tokens=256,
            stream=False
        )
        hist.append({"role":"assistant","content":conv})
        return hist, ""

    tool = call.get("tool")
    if tool=="metadata":
        reply = run_metadata(file)
    elif tool=="count_classes":
        reply = run_count_classes(file)
    elif tool=="list_metrics":
        reply = run_list_metrics(file)
    elif tool=="compute_metrics":
        raw_metrics = call.get("metrics", [])
        level       = call.get("level", "both")
        # map & compute
        mapped, unknowns = [], []
        for m in raw_metrics:
            ml = m.lower()
            if ml in metric_definitions:
                mapped.append(ml)
            elif ml in reverse_synonyms:
                mapped.append(reverse_synonyms[ml])
            else:
                unknowns.append(m)
        if unknowns:
            reply = f"Sorry, I don’t recognize: {', '.join(unknowns)}. Could you clarify?"
        else:
            reply = run_compute_metrics(file, mapped, level)
    else:
        reply = "Sorry, I didn’t understand that. Could you clarify?"

    hist.append({"role":"assistant","content":reply})
    return hist, ""

# --- UI setup & launch ---
initial_history = [{"role":"assistant","content":
                   "👋 Hi! I’m Spatchat. Upload a GeoTIFF to begin—then ask for any landscape metric."}]

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
            clr_chat = gr.Button("Clear Chat")

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
    clr_chat.click(lambda: initial_history, outputs=chatbot)

iface.launch(ssr_mode=False)