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
import json

# --- LLM setup ---
load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# --- Metric definitions ---
metric_definitions = {
    "pland":        ("Proportion of Landscape (PLAND)", "Percentage of landscape comprised by the class."),
    "np":           ("Number of Patches (NP)", "Total number of patches for the class or entire landscape."),
    "pd":           ("Patch Density (PD)", "Number of patches per 100 hectares."),
    "lpi":          ("Largest Patch Index (LPI)", "Percentage of total landscape made up by the largest patch."),
    "te":           ("Total Edge (TE)", "Total length of all patch edges."),
    "edge_density": ("Edge Density (ED)", "Edge length per hectare."),
    "lsi":          ("Landscape Shape Index (LSI)", "Overall shape complexity of the landscape."),
    "tca":          ("Total Core Area (TCA)", "Sum of all core areas in the landscape."),
    "mesh":         ("Effective Mesh Size (MESH)", "Average patch size accounting for edges."),
    "contag":       ("Contagion Index (CONTAG)", "Clumpiness of patches."),
    "shdi":         ("Shannon Diversity Index (SHDI)", "Diversity of patch types."),
    "shei":         ("Shannon Evenness Index (SHEI)", "Evenness of patch distribution."),
    "area":         ("Total Area (AREA)", "Total area of patches by class or landscape."),
    "perim":        ("Total Perimeter (PERIM)", "Total perimeter of all patches."),
    "para":         ("Perimeter-Area Ratio (PARA)", "Ratio of perimeter to area."),
    "shape":        ("Shape Index (SHAPE)", "Normalized perimeter-area ratio."),
    "frac":         ("Fractal Dimension (FRAC)", "Fractal complexity of patch shapes."),
    "enn":          ("Euclidean Nearest Neighbor (ENN)", "Distance to nearest patch of same class."),
    "core":         ("Total Core Area (CORE)", "Sum of interior (nondisturbed) patch area."),
    "nca":          ("Number of Core Areas (NCA)", "Count of core regions in the landscape."),
    "cai":          ("Core Area Index (CAI)", "Proportion of core area to total patch area."),
}

# --- Synonyms for query matching ---
synonyms = {
    "edge_density": ["ed", "edge density"],
    "pland": ["pland", "proportion of landscape"],
    "np":    ["np", "number of patches"],
    "pd":    ["pd", "patch density"],
    "lpi":   ["lpi", "largest patch"],
    "te":    ["te", "total edge"],
}

# --- Column mapping to pylandstats names ---
col_map = {
    "pland":        "proportion_of_landscape",
    "np":           "number_of_patches",
    "pd":           "patch_density",
    "lpi":          "largest_patch_index",
    "te":           "total_edge",
    "edge_density": "edge_density",
    "lsi":          "landscape_shape_index",
    "tca":          "total_core_area",
    "mesh":         "effective_mesh_size",
    "contag":       "contagion_index",
    "shdi":         "shannon_diversity_index",
    "shei":         "shannon_evenness_index",
    "area":         "total_area",
    "perim":        "total_perimeter",
    "para":         "perimeter_area_ratio",
    "shape":        "shape_index",
    "frac":         "fractal_dimension_index",
    "enn":          "euclidean_nearest_neighbor_distance",
    "core":         "total_core_area",
    "nca":          "number_of_core_areas",
    "cai":          "core_area_index",
}

# --- Level categories ---
helper_methods = {
    "pd":           "patch_density",
    "edge_density": "edge_density",
    "lsi":          "landscape_shape_index",
    "contag":       "contagion_index",
    "shdi":         "shannon_diversity_index",
    "shei":         "shannon_evenness_index",
    "mesh":         "effective_mesh_size",
    "tca":          "total_core_area",
    "lpi":          "largest_patch_index",
    "te":           "total_edge",
}
class_only = {"pland","area","perim","para","shape","frac","enn","core","nca","cai"}
cross_level = ["np","pd","lpi","te","edge_density"]

# --- Raster preview & clear helpers ---
def no_raster_fig():
    fig, ax = plt.subplots(figsize=(5,5))
    ax.text(0.5,0.5,"🗂️ No raster loaded.", ha='center', va='center', color='gray', fontsize=14)
    ax.set_title("Raster Preview", color='dimgray')
    ax.axis('off')
    return fig

def preview_raster(file):
    try:
        if not file:
            return no_raster_fig()
        with rasterio.open(file.name) as src:
            raw = src.read(1)
            nodata = src.nodata or 0
        if raw.dtype.kind in ('U','S','O'):
            data_str = raw.astype(str)
            uniq = np.unique(data_str[data_str!=""])
            arr = np.zeros_like(raw, dtype=int)
            for i, nm in enumerate(uniq,1):
                arr[data_str==nm] = i
            labels = [f"{i}: {nm}" for i,nm in enumerate(uniq,1)]
        else:
            arr = raw
            vals = np.unique(arr[arr!=nodata])
            labels = [f"Class {int(v)}" for v in vals]
        n = len(labels)
        colors = plt.cm.tab10(np.linspace(0,1,n))
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(arr, cmap='tab10', interpolation='nearest')
        ax.set_title("Uploaded Raster")
        ax.axis('off')
        handles = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(n)]
        ax.legend(handles=handles, loc='lower left', fontsize='small', frameon=True)
        return fig
    except Exception:
        return no_raster_fig()

def clear_raster():
    return None, gr.update(value=no_raster_fig(), visible=True)

def notify_upload(file, history):
    if file:
        return history + [{"role":"assistant","content":"📥 Raster uploaded successfully!"}], ""
    return history, ""

# --- Core handlers ---
def answer_metadata(file, history):
    with rasterio.open(file.name) as src:
        crs, x_res, y_res = src.crs, *src.res
        extent, bands, nodata = src.bounds, src.count, src.nodata
        unit = getattr(src.crs, 'linear_units', 'unit')
    text = f"CRS: {crs}\nResolution: {x_res:.2f}×{y_res:.2f} {unit}\nExtent: {extent}\nBands: {bands}\nNoData: {nodata}"
    return history + [{"role":"assistant","content":text}], ""

def count_classes(file, history):
    """
    Counts unique classes using pylandstats to ensure all present classes are captured, bypassing nodata issues.
    """
    # Build the landscape object (handles nodata internally)
    ls = _build_landscape(file)
    # Compute class metrics & count rows
    df = ls.compute_class_metrics_df(metrics=["number_of_patches"]).rename_axis("code").reset_index()
    num = df.shape[0]
    return history + [{"role":"assistant","content":f"Your raster contains {num} unique classes."}], {"role":"assistant","content":f"Your raster contains {len(vals)} unique classes."}], ""

def list_metrics(history):
    lines = ["**Cross‑level metrics (Landscape & Class):**"]
    for k in cross_level:
        lines.append(f"- {metric_definitions[k][0]} (`{k}`)")
    lines.append("\n**Landscape‑only metrics:**")
    for k in helper_methods:
        lines.append(f"- {metric_definitions[k][0]} (`{k}`)")
    lines.append("\n**Class‑only metrics:**")
    for k in class_only:
        lines.append(f"- {metric_definitions[k][0]} (`{k}`)")
    return history + [{"role":"assistant","content":"\n".join(lines)}], ""

def compute_landscape_only(file, keys, history):
    ls = Landscape(file.name, nodata=getattr(rasterio.open(file.name), 'nodata', 0), res=tuple(rasterio.open(file.name).res))
    parts = []
    for key in keys:
        name, _ = metric_definitions[key]
        if key == "np":
            df_tmp = ls.compute_class_metrics_df(metrics=["number_of_patches"]);
            val = int(df_tmp["number_of_patches"].sum())
        elif key in helper_methods:
            val = getattr(ls, helper_methods[key])()
        else:
            df_land = ls.compute_landscape_metrics_df(metrics=[col_map[key]]);
            val = df_land[col_map[key]].iloc[0]
        parts.append(f"**{name} ({key.upper()}):** {val:.4f}" if isinstance(val, float) else f"**{name} ({key.upper()}):** {val}")
    return history + [{"role":"assistant","content":"\n\n".join(parts)}], ""


def compute_class_only(file, keys, history):
    ls = Landscape(file.name, nodata=getattr(rasterio.open(file.name), 'nodata', 0), res=tuple(rasterio.open(file.name).res))
    cols = [col_map[k] for k in keys]
    df2 = ls.compute_class_metrics_df(metrics=cols).rename_axis("code").reset_index()
    df2["class_name"] = df2["code"].map(lambda c: f"Class {int(c)}")
    tbl = df2[["class_name"] + cols].to_markdown(index=False)
    return history + [{"role":"assistant","content":f"**Class-level metrics:**\n{tbl}"}], ""

def compute_multiple_metrics(file, keys, history):
    hl, _ = compute_landscape_only(file, keys, history)
    hc, _ = compute_class_only(file, keys, history)
    return history + [hl[-1], hc[-1]], ""

def llm_fallback(history):
    prompt = [{"role":"system","content":"You are Spatchat, help with landscape metrics."}, *history]
    resp = client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", messages=prompt, temperature=0.4).choices[0].message.content
    return history + [{"role":"assistant","content":resp}], ""

# --- Main handler ---
def analyze_raster(file, question, history):
    history, _ = notify_upload(file, history)
    hist = history + [{"role":"user","content":question}]
    # parse intents with LLM
    parsed = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {"role":"system","content":"Parse into JSON keys: list_metrics,count_classes,metadata,metrics,level,all_metrics"},
            {"role":"user","content":question}
        ],
        temperature=0.0
    ).choices[0].message.content
    try:
        req = json.loads(parsed)
    except:
        req = {}
    if req.get("list_metrics"):  return list_metrics(hist)
    if req.get("count_classes"): return count_classes(file, hist)
    if req.get("metadata"):      return answer_metadata(file, hist)
    if not file: return hist + [{"role":"assistant","content":"Please upload a GeoTIFF before asking anything."}], ""
    metrics = req.get("metrics", [])
    if req.get("all_metrics"):
        if req.get("level") == "landscape": metrics = [m for m in metric_definitions if m not in class_only]
        elif req.get("level") == "class": metrics = list(class_only)
    if not metrics:
        for code in metric_definitions:
            if re.search(rf"\b{re.escape(code)}\b", question.lower()): metrics.append(code)
    level = req.get("level", "both")
    if not metrics: return llm_fallback(hist)
    if level == "landscape": return compute_landscape_only(file, metrics, hist)
    if level == "class":     return compute_class_only(file, metrics, hist)
    return compute_multiple_metrics(file, metrics, hist)


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
