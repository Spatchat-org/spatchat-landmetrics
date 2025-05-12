import os
import re
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

# --- Synonyms ---
synonyms = {
    "edge_density": ["ed", "edge density"],
    "pland":        ["pland", "proportion of landscape"],
    "np":           ["np", "number of patches"],
    "pd":           ["pd", "patch density"],
    "lpi":          ["lpi", "largest patch index"],
    "te":           ["te", "total edge"],
}

# --- Column mapping ---
col_map = {
    "pland": "proportion_of_landscape",
    "np": "number_of_patches",
    "pd": "patch_density",
    "lpi": "largest_patch_index",
    "te": "total_edge",
    "edge_density": "edge_density",
    "lsi": "landscape_shape_index",
    "tca": "total_core_area",
    "mesh": "effective_mesh_size",
    "contag": "contagion_index",
    "shdi": "shannon_diversity_index",
    "shei": "shannon_evenness_index",
    "area": "total_area",
    "perim": "total_perimeter",
    "para": "perimeter_area_ratio",
    "shape": "shape_index",
    "frac": "fractal_dimension_index",
    "enn": "euclidean_nearest_neighbor_distance",
    "core": "total_core_area",
    "nca": "number_of_core_areas",
    "cai": "core_area_index",
}

# --- Level categories ---
helper_methods = {
    "pd": "patch_density",
    "edge_density": "edge_density",
    "lsi": "landscape_shape_index",
    "contag": "contagion_index",
    "shdi": "shannon_diversity_index",
    "shei": "shannon_evenness_index",
    "mesh": "effective_mesh_size",
    "tca": "total_core_area",
    "lpi": "largest_patch_index",
    "te": "total_edge",
}
class_only = {"pland", "area", "perim", "para", "shape", "frac", "enn", "core", "nca", "cai"}
cross_level = ["np", "pd", "lpi", "te", "edge_density"]

# --- Raster preview ---
def no_raster_fig():
    fig, ax = plt.subplots(figsize=(5,5))
    ax.text(0.5, 0.5, "🗂️ No raster loaded.", ha='center', va='center', color='gray', fontsize=14)
    ax.set_title("Raster Preview", color='dimgray')
    ax.axis('off')
    return fig

def preview_raster(file):
    if not file:
        return no_raster_fig()
    with rasterio.open(file.name) as src:
        raw = src.read(1)
        nodata = src.nodata or 0
    if raw.dtype.kind in ('U','S','O'):
        data = raw.astype(str)
        uniq = np.unique(data[data!=""])
        arr = np.zeros_like(raw, dtype=int)
        for i, nm in enumerate(uniq, 1):
            arr[data == nm] = i
        labels = [f"{i}: {nm}" for i, nm in enumerate(uniq, 1)]
    else:
        arr = raw
        vals = np.unique(arr[arr != nodata])
        labels = [f"Class {int(v)}" for v in vals]
    n = len(labels)
    colors = plt.cm.tab10(np.linspace(0,1,n))
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(arr, cmap='tab10')
    ax.set_title("Uploaded Raster")
    ax.axis('off')
    handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=handles, loc='lower left', fontsize='small')
    return fig

def clear_raster():
    return None, gr.update(value=no_raster_fig(), visible=True)

def notify_upload(file, history):
    if file:
        return history + [{"role": "assistant", "content": "📥 Raster uploaded successfully!"}], ""
    return history, ""

# --- Handlers ---
def answer_metadata(file, history):
    with rasterio.open(file.name) as src:
        text = (
            f"CRS: {src.crs}\n"
            f"Resolution: {src.res[0]:.2f}×{src.res[1]:.2f}\n"
            f"Extent: {src.bounds}\n"
            f"Bands: {src.count}\n"
            f"NoData: {src.nodata}"
        )
    return history + [{"role": "assistant", "content": text}], ""

def count_classes(file, history):
    with rasterio.open(file.name) as src:
        arr = src.read(1)
        nodata = src.nodata or 0
    vals = np.unique(arr[arr != nodata])
    return history + [{"role": "assistant", "content": f"Your raster contains {len(vals)} unique classes."}], ""

def list_metrics(history):
    lines = ["**Cross‑level metrics:**"]
    lines += [f"- {metric_definitions[k][0]} (`{k}`)" for k in cross_level]
    lines.append("\n**Landscape‑only:**")
    lines += [f"- {metric_definitions[k][0]} (`{k}`)" for k in helper_methods]
    lines.append("\n**Class‑only:**")
    lines += [f"- {metric_definitions[k][0]} (`{k}`)" for k in class_only]
    return history + [{"role": "assistant", "content": "\n".join(lines)}], ""

def compute_landscape_only(file, keys, history):
    ls = Landscape(file.name, nodata=0, res=(1,1))
    parts = []
    for key in keys:
        name = metric_definitions[key][0]
        if key == "np":
            df = ls.compute_class_metrics_df(metrics=["number_of_patches"])
            val = int(df.number_of_patches.sum())
        elif key in helper_methods:
            val = getattr(ls, helper_methods[key])()
        else:
            val = ls.compute_landscape_metrics_df(metrics=[col_map[key]]).iloc[0, 0]
        parts.append(f"**{name}:** {val:.4f}" if isinstance(val, float) else f"**{name}:** {val}")
    return history + [{"role": "assistant", "content": "\n\n".join(parts)}], ""

def compute_class_only(file, keys, history):
    """
    Compute *only* the class‐level metrics for the given keys.
    Returns (new_history, ""), where new_history is the chat history list.
    """
    ls   = _build_landscape(file)
    cols = [col_map[k] for k in keys]
    # get a DataFrame of class‐level metrics
    df   = (
        ls
        .compute_class_metrics_df(metrics=cols)
        .rename_axis("code")
        .reset_index()
    )
    # build friendly class names
    df["class_name"] = df["code"].astype(int).apply(lambda c: f"Class {c}")

    # reorder into [class_name, metric1, metric2, ...]
    out_cols = ["class_name"] + cols
    tbl      = df[out_cols].to_markdown(index=False)

    content  = f"**Class-level metrics:**\n{tbl}"
    return history + [{"role":"assistant","content":content}], ""


def compute_multiple_metrics(file, keys, history):
    """
    Compute both landscape- and class-level metrics for the given keys.
    Landscape-level is only run on keys *not* in class_only, then
    class-level is run on *all* keys.
    """
    # split your keys
    landscape_keys = [k for k in keys if k not in class_only]
    class_keys     = keys

    # 1️⃣ run landscape-only on the filtered set (if any)
    chat, _ = (
        compute_landscape_only(file, landscape_keys, history)
        if landscape_keys
        else (history, "")
    )

    # 2️⃣ run class-only on the full set, appending to the same chat
    chat, _ = compute_class_only(file, class_keys, chat)

    return chat, ""


def llm_fallback(history):
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[{"role": "system", "content": "You are Spatchat, assist with landscape metrics."}] + history,
        temperature=0.4
    ).choices[0].message.content
    return history + [{"role": "assistant", "content": resp}], ""

# --- Main handler (shortcut) ---
def analyze_raster(file, question, history):
    history, _ = notify_upload(file, history)
    lower = question.lower()

    if re.search(r"\b(list|available).*metrics\b", lower):
        return list_metrics(history)
    if not file:
        return history + [{"role": "assistant", "content": "Please upload a GeoTIFF first."}], ""
    if re.search(r"how many classes", lower):
        return count_classes(file, history)
    if re.search(r"\b(crs|resolution|extent|bands|nodata)\b", lower):
        return answer_metadata(file, history)

    # find metrics
    found = []
    for code in metric_definitions:
        for syn in synonyms.get(code, [code]):
            if syn in lower and code not in found:
                found.append(code)

    if not found:
        return llm_fallback(history)

    # dispatch
    if len(found) > 1:
        return compute_multiple_metrics(file, found, history)
    # single metric: decide level
    if any((f == "np") or (f in helper_methods) for f in found):
        return compute_landscape_only(file, found, history)
    else:
        return compute_class_only(file, found, history)


# --- UI setup & launch ---
initial_history = [{"role":"assistant","content":"👋 Hi! I’m Spatchat. Upload a GeoTIFF to begin—then ask for any landscape metric."}]

with gr.Blocks(title="Spatchat") as iface:
    gr.HTML('<head><link rel="icon" href="logo1.png"></head>')
    gr.Image(value="logo/logo_long1.png", type="filepath", show_label=False, show_download_button=False, show_share_button=False, elem_id="logo-img")
    gr.HTML("<style>#logo-img img{height:90px;margin:10px;border-radius:6px;}</style>")
    gr.Markdown("## 🌲 Spatchat: Landscape Metrics Assistant")
    gr.HTML('''
      <div style="margin:-10px 0 20px;">
        <input id="shareLink" type="text" readonly
               value="https://spatchat.org/browse/?room=landmetrics"
               style="width:50%;padding:5px;border:1px solid #ccc;border-radius:4px;" />
        <button onclick="navigator.clipboard.writeText(
          document.getElementById('shareLink').value
        )" style="padding:5px 10px;background:#007BFF;color:white;border:none;border-radius:4px;">
          📋 Copy Share Link
        </button>
      </div>
    ''')
    
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
