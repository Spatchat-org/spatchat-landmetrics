import os
import re
import json
import gradio as gr
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pylandstats import Landscape
from together import Together
from dotenv import load_dotenv

# --- LLM setup ---
load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

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

# --- Synonyms (if needed later) ---
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

# --- Raster preview helpers (unchanged) ---
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

# --- Core metric / metadata helpers (unchanged) ---
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
    lines = ["**Cross‑level metrics:**"]
    lines += [f"- {metric_definitions[k][0]} (`{k}`)" for k in cross_level]
    lines.append("\n**Landscape‑only:**")
    lines += [f"- {metric_definitions[k][0]} (`{k}`)" for k in helper_methods]
    lines.append("\n**Class‑only:**")
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
    parts = []
    for key in keys:
        name,_ = metric_definitions[key]
        if key=="np":
            df = ls.compute_class_metrics_df(metrics=["number_of_patches"])
            val= int(df["number_of_patches"].sum())
        elif key in helper_methods:
            val = getattr(ls, helper_methods[key])()
        else:
            df  = ls.compute_landscape_metrics_df(metrics=[col_map[key]])
            val = df[col_map[key]].iloc[0]
        parts.append(
            f"**{name} ({key.upper()}):** {val:.4f}"
            if isinstance(val,float)
            else f"**{name} ({key.upper()}):** {val}"
        )
    return "\n\n".join(parts)

def compute_class_only_text(file, keys):
    ls   = _build_landscape(file)
    cols = [col_map[k] for k in keys]
    df   = ls.compute_class_metrics_df(metrics=cols).rename_axis("code").reset_index()
    df["class_name"] = df["code"].astype(int).apply(lambda c:f"Class {c}")
    tbl  = df[["class_name"] + cols].to_markdown(index=False)
    return f"**Class-level metrics:**\n{tbl}"

def compute_multiple_metrics_text(file, keys):
    # split into pure‑landscape vs class‑only
    land_keys  = [c for c in keys if c not in class_only]
    class_keys = [c for c in keys if c in class_only]

    out = []
    if land_keys:
        out.append(compute_landscape_only_text(file, land_keys))
    if class_keys:
        out.append(compute_class_only_text(file, class_keys))

    return "\n\n".join(out)

# ───── Your “tools” ─────────────────────────────────────────────────────

def run_metadata(file):
    return answer_metadata(file)

def run_count_classes(file):
    return count_classes(file)

def run_list_metrics(file):
    return list_metrics_text()

def run_compute_metrics(file, raw_metrics, level):
    # 1) normalize + map synonyms & reject any None‑mapped code
    mapped = []
    unknown = []
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
            # e.g. "shei"
            unknown.append(m)
        else:
            mapped.append(cand)
    if unknown:
        return f"Sorry, I don’t recognize: {', '.join(unknown)}. Could you clarify?"

    # 2) if the user explicitly asked "landscape" or "class", honor that
        land = [c for c in mapped if c not in class_only]
        return compute_landscape_only_text(file, land)

    if level == "class":
        clas = [c for c in mapped if c in class_only]
        return compute_class_only_text(file, clas)

    # 3) otherwise infer from the mix of codes
    has_x = any(c in cross_level for c in mapped)
    has_c = any(c in class_only   for c in mapped)

    # a) mixed → both
    if has_x and has_c:
        return compute_multiple_metrics_text(file, mapped)

    # b) only cross‑level → landscape only
    if has_x:
        land = [c for c in mapped if c in cross_level]
        return compute_landscape_only_text(file, land)

    # c) only class‑level → class only
    return compute_class_only_text(file, mapped)


# ───── System & Fallback Prompts ────────────────────────────────────────

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
You are Spatchat, a landscape‑metrics expert.  
Keep replies under two sentences.  
If you can’t map a request to one of your tools, ask the user to clarify.
""".strip()

# ───── Main handler ─────────────────────────────────────────────────────

def analyze_raster(file, user_msg, history):
    hist = history + [{"role":"user","content":user_msg}]
    # 1️⃣ Ensure there's a raster
    if file is None:
        hist.append({"role":"assistant","content":
                     "Please upload a GeoTIFF before asking anything."})
        return hist, ""

    # ─── Zero‑prompt shortcuts ───────────────────────────────────────────────
    # “all landscape level metrics”
    if re.search(r"\ball landscape(?: level)? metrics\b", lower):
        keys = [k for k in metric_map if metric_map.get(k) and k not in class_only]
        return hist + [{"role":"assistant","content":compute_landscape_only_text(file, keys)}], ""

    # “all class level metrics”
    if re.search(r"\ball class(?: level)? metrics\b", lower):
        keys = list(class_only)
        return hist + [{"role":"assistant","content":compute_class_only_text(file, keys)}], ""

    # “all metrics”
    if re.search(r"\ball metrics\b", lower):
        keys = [k for k in metric_map if metric_map.get(k)]
        return hist + [{"role":"assistant","content":compute_multiple_metrics_text(file, keys)}], ""

    # 2️⃣ Ask the LLM which tool to call
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role":"system","content":SYSTEM_PROMPT}] + hist,
        temperature=0.0
    ).choices[0].message.content

    # 3️⃣ Parse JSON
    try:
        call = json.loads(resp)
    except:
        # fallback to conversation
        conv = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role":"system","content":FALLBACK_PROMPT}] + hist,
            temperature=0.7
        ).choices[0].message.content
        hist.append({"role":"assistant","content":conv})
        return hist, ""

    tool = call.get("tool")
    if tool=="metadata":
        reply = run_metadata(file)
    elif tool=="count_classes":
        reply = run_count_classes(file)
    elif tool=="list_metrics":
        reply = run_list_metrics(file)
    elif tool == "compute_metrics":
        raw_metrics = call.get("metrics", [])
        level       = call.get("level", "both")

        # 1️⃣ normalize + map synonyms → canonical codes
        mapped   = []
        unknowns = []
        for m in raw_metrics:
            ml = m.lower()
            if ml in metric_definitions:
                mapped.append(ml)
            elif ml in reverse_synonyms:
                mapped.append(reverse_synonyms[ml])
            else:
                unknowns.append(m)

        # 2️⃣ if any still unknown, ask to clarify
        if unknowns:
            reply = f"Sorry, I don’t recognize: {', '.join(unknowns)}. Could you clarify?"
        else:
            # 3️⃣ compute with canonical codes
            reply = run_compute_metrics(file, mapped, level)
    else:
        reply = "Sorry, I didn’t understand that. Could you clarify?"

    hist.append({"role":"assistant","content":reply})
    return hist, ""

# --- UI setup & launch ---
initial_history = [{"role":"assistant","content":"👋 Hi! I’m Spatchat. Upload a GeoTIFF to begin—then ask for any landscape metric."}]

with gr.Blocks(title="Spatchat") as iface:
    gr.HTML('<head><link rel="icon" href="logo1.png"></head>')
    gr.Image(value="logo_long1.png", type="filepath", show_label=False, show_download_button=False, show_share_button=False, elem_id="logo-img")
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
    gr.Markdown("""
                <div style="font-size: 14px;">
                © 2025 Ho Yi Wan & Logan Hysen. All rights reserved.<br>
                If you use Spatchat in research, please cite:<br>
                <b>Wan, H.Y.</b> & <b>Hysen, L.</b> (2025). <i>Spatchat: Landscape Metrics.</i>
                </div>
                """)
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload GeoTIFF", type="filepath")
            raster_out = gr.Plot(label="Raster Preview")
            clear_btn = gr.Button("Clear Raster")
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(value=initial_history, type="messages", label="Spatchat Dialog")
            txt_in = gr.Textbox(placeholder="e.g. calculate edge density", lines=1)
            clr_chat = gr.Button("Clear Chat")

    file_input.change(preview_raster, inputs=file_input, outputs=raster_out)
    file_input.change(notify_upload, inputs=[file_input, chatbot], outputs=[chatbot, txt_in])
    clear_btn.click(clear_raster, inputs=None, outputs=[file_input, raster_out])
    txt_in.submit(analyze_raster, inputs=[file_input, txt_in, chatbot], outputs=[chatbot, txt_in])
    clr_chat.click(lambda: initial_history, outputs=chatbot)

iface.launch()
