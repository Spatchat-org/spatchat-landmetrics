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

# --- Metric glossary (hard‑coded) ---
metric_definitions = {
    "pland":        ("Proportion of Landscape (PLAND)",      "Percentage of landscape comprised by the class."),
    "np":           ("Number of Patches (NP)",               "Total number of patches for the class or entire landscape."),
    "pd":           ("Patch Density (PD)",                   "Number of patches per 100 hectares."),
    "lpi":          ("Largest Patch Index (LPI)",            "Percentage of total landscape made up by the largest patch."),
    "te":           ("Total Edge (TE)",                      "Total length of all patch edges."),
    "edge_density": ("Edge Density (ED)",                    "Edge length per hectare."),
    "lsi":          ("Landscape Shape Index (LSI)",          "Overall shape complexity of the landscape."),
    "tca":          ("Total Core Area (TCA)",                "Sum of all core areas in the landscape."),
    "mesh":         ("Effective Mesh Size (MESH)",           "Average patch size accounting for edges."),
    "contag":       ("Contagion Index (CONTAG)",             "Clumpiness of patches."),
    "shdi":         ("Shannon Diversity Index (SHDI)",       "Diversity of patch types."),
    "shei":         ("Shannon Evenness Index (SHEI)",        "Evenness of patch distribution."),
    "area":         ("Total Area (AREA)",                    "Total area of patches by class or landscape."),
    "perim":        ("Total Perimeter (PERIM)",               "Total perimeter of all patches."),
    "para":         ("Perimeter-Area Ratio (PARA)",          "Ratio of perimeter to area."),
    "shape":        ("Shape Index (SHAPE)",                  "Normalized perimeter-area ratio."),
    "frac":         ("Fractal Dimension (FRAC)",             "Fractal complexity of patch shapes."),
    "enn":          ("Euclidean Nearest Neighbor (ENN)",     "Distance to nearest patch of same class."),
    "core":         ("Total Core Area (CORE)",               "Sum of interior (nondisturbed) patch area."),
    "nca":          ("Number of Core Areas (NCA)",           "Count of core regions in the landscape."),
    "cai":          ("Core Area Index (CAI)",                "Proportion of core area to total patch area."),
}

# --- Synonyms for natural names ---
synonyms = {
    "edge_density": ["ed", "edge density", "edge_density"],
    "pland":        ["pland", "proportion of landscape"],
    "np":           ["np", "number of patches"],
    "pd":           ["pd", "patch density"],
    "lpi":          ["lpi", "largest patch"],
    "te":           ["te", "total edge", "total_edge"],
}

# --- Map codes to DataFrame columns ---
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

# --- Helper mappings & level categorization ---
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

# Metrics strictly available only at class level:
class_only = {
    "pland",
    "area", "perim", "para", "shape", "frac",
    "enn", "core", "nca", "cai",
}

# Metrics available at both landscape & class levels:
cross_level = ["np", "pd", "lpi", "te", "edge_density"]

# --- Raster preview ---
def no_raster_fig():
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.text(0.5, 0.5, "🗂️ No raster loaded.", ha='center', va='center', color='gray', fontsize=14)
    ax.set_title("Raster Preview", color='dimgray')
    ax.axis('off')
    return fig

def preview_raster(file):
    try:
        if file is None:
            return no_raster_fig()
        with rasterio.open(file.name) as src:
            raw    = src.read(1)
            nodata = src.nodata or 0

        # string‐typed?
        if raw.dtype.kind in ('U','S','O'):
            data_str     = raw.astype(str)
            unique_names = np.unique(data_str[data_str != ""])
            name2code    = {nm: i+1 for i, nm in enumerate(unique_names)}
            data         = np.zeros_like(raw, dtype=int)
            for nm, code in name2code.items():
                data[data_str == nm] = code
            labels = [f"{i+1}: {unique_names[i]}" for i in range(len(unique_names))]
            n = len(unique_names)
        else:
            data = raw
            vals = np.unique(data[data != nodata])
            labels = [f"Class {int(v)}" for v in vals]
            n = len(vals)

        colors = plt.cm.tab10(np.linspace(0,1,n))
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(data, cmap='tab10', interpolation='nearest', vmin=1, vmax=n)
        ax.set_title("Uploaded Raster")
        ax.axis('off')
        handles = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(n)]
        ax.legend(handles=handles, loc='lower left', fontsize='small', frameon=True)
        return fig
    except Exception as e:
        print("preview_raster error:", e)
        return no_raster_fig()

def clear_raster():
    fig = no_raster_fig()
    return None, gr.update(value=fig, visible=True)

# --- Handlers ---
def answer_metadata(file, history):
    with rasterio.open(file.name) as src:
        crs     = src.crs
        x_res,y_res = src.res
        extent  = src.bounds
        bands   = src.count
        nodata  = src.nodata
        unit    = getattr(src.crs, "linear_units", "unit")
    text = "\n".join([
        f"CRS: {crs}",
        f"Resolution: {x_res:.2f} × {y_res:.2f} {unit}",
        f"Extent: {extent}",
        f"Bands: {bands}",
        f"NoData value: {nodata}" 
    ])
    return history + [{"role":"assistant","content":text}], ""

def list_metrics(history):
    lines = ["**Cross‑level metrics (Landscape & Class):**"]
    for k in cross_level:
        name, _ = metric_definitions[k]
        lines.append(f"- {name} (`{k}`)")
    lines.append("\n**Landscape‑only metrics:**")
    for k in helper_methods:
        name, _ = metric_definitions[k]
        lines.append(f"- {name} (`{k}`)")
    lines.append("\n**Class‑only metrics:**")
    for k in class_only:
        name, _ = metric_definitions[k]
        lines.append(f"- {name} (`{k}`)")
    content = "\n".join(lines)
    return history + [{"role":"assistant","content":content}], ""

# --- Compute helpers ---
def _build_landscape(file):
    with rasterio.open(file.name) as src:
        raw = src.read(1)
        x_res, y_res = src.res
        nodata = src.nodata or 0
    if raw.dtype.kind in ("U","S","O"):
        data_str = raw.astype(str)
        uniq = np.unique(data_str[data_str!=""])
        name2code = {nm: i+1 for i,nm in enumerate(uniq)}
        arr = np.zeros_like(raw, dtype=int)
        for nm, code in name2code.items():
            arr[data_str==nm] = code
        ls = Landscape(arr, res=(x_res,y_res), nodata=0)
    else:
        ls = Landscape(file.name, nodata=nodata, res=(x_res,y_res))
    return ls

def compute_landscape_only(file, keys, history):
    ls = _build_landscape(file)
    parts = []
    for key in keys:
        name, _ = metric_definitions[key]
        if key == "np":
            df = ls.compute_class_metrics_df(metrics=["number_of_patches"])
            val = int(df["number_of_patches"].sum())
        elif key in helper_methods:
            val = getattr(ls, helper_methods[key])()
        else:
            df = ls.compute_landscape_metrics_df(metrics=[col_map[key]])
            val = df[col_map[key]].iloc[0]
        parts.append(f"**{name} ({key.upper()}):** {val:.4f}" if isinstance(val, float) else f"**{name} ({key.upper()}):** {val}")
    content = "\n\n".join(parts)
    return history + [{"role":"assistant","content":content}], ""

def compute_class_only(file, keys, history):
    ls = _build_landscape(file)
    cols = [col_map[k] for k in keys]
    df = ls.compute_class_metrics_df(metrics=cols).rename_axis("code").reset_index()
    df["class_name"] = df["code"].map(lambda c: f"Class {int(c)}")
    tbl = df[["class_name"] + cols].to_markdown(index=False)
    content = f"**Class-level metrics:**\n{tbl}"
    return history + [{"role":"assistant","content":content}], ""

def compute_multiple_metrics(file, keys, history):
    hl, _ = compute_landscape_only(file, keys, history)
    hc, _ = compute_class_only(file, keys, history)
    return history + [hl[-1], hc[-1]], ""

# --- Main handler ---
def analyze_raster(file, question, history):
    hist = history + [{"role":"user","content":question}]
    lower = question.lower()

    # 1️⃣ Parse all intents using LLM
    parse = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {"role":"system","content":(
                "Parse the user request into JSON with these fields:\n"
                "- list_metrics: true/false\n"
                "- count_classes: true/false\n"
                "- metadata: true/false\n"
                "- metrics: [list of metric codes]\n"
                "- level: 'landscape', 'class', or 'both'\n"
                "- all_metrics: true/false\n"
                "Output only valid JSON."
            )},
            {"role":"user","content":question}
        ],
        temperature=0.0
    ).choices[0].message.content

    try:
        req = json.loads(parse)
    except json.JSONDecodeError:
        req = {}

    # 2️⃣ Handle simple list/metadata/class-count intents
    if req.get("list_metrics"):
        return list_metrics(hist)
    if req.get("count_classes"):
        return count_classes(file, hist)
    if req.get("metadata"):
        return answer_metadata(file, hist)

    # 3️⃣ Ensure GeoTIFF is loaded for any metric work
    if file is None:
        return hist + [{"role":"assistant","content":
                        "Please upload a GeoTIFF before asking anything."}], ""

    # 4️⃣ Build metric list
    metrics = req.get("metrics", [])
    level   = req.get("level",   "both")
    all_m   = req.get("all_metrics", False)

    # 5️⃣ Expand 'all_metrics' if requested
    if all_m:
        if level == "landscape":
            metrics = [m for m in metric_definitions if m not in class_only]
        elif level == "class":
            metrics = list(class_only)

    # 6️⃣ If no metrics at this point, fallback
    if not metrics:
        return llm_fallback(hist)

    # 7️⃣ Dispatch to the correct helper
    if level == "landscape":
        return compute_landscape_only(file, metrics, hist)
    if level == "class":
        return compute_class_only(file, metrics, hist)
    return compute_multiple_metrics(file, metrics, hist)

# --- UI layout & launch ---
initial_history = [
    {"role":"assistant","content":
     "👋 Hi! I’m Spatchat. Upload a GeoTIFF to begin—then ask for CRS, resolution, extent, or any landscape metric."}
]

with gr.Blocks(title="Spatchat") as iface:
    gr.HTML('<head><link rel="icon" href="file=logo1.png"></head>')
    gr.Image(value="logo/logo_long1.png", type="filepath",
             show_label=False, show_download_button=False,
             show_share_button=False, elem_id="logo-img")
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
    '''
)
    with gr.Row():
        with gr.Column(scale=1):
            file_input          = gr.File(label="Upload GeoTIFF", type="filepath")
            raster_output       = gr.Plot(label="Raster Preview")
            clear_raster_button = gr.Button("Clear Raster")
        with gr.Column(scale=1):
            chatbot        = gr.Chatbot(value=initial_history, type="messages",
                                       label="Spatchat Dialog")
            question_input = gr.Textbox(placeholder="e.g. calculate edge density",\n                                       lines=1)
            clear_button   = gr.Button("Clear Chat")
)
    file_input.change(preview_raster, inputs=file_input, outputs=raster_output)
    clear_raster_button.click(clear_raster, inputs=None,
                              outputs=[file_input, raster_output])
    question_input.submit(analyze_raster,
                          inputs=[file_input, question_input, chatbot],
                          outputs=[chatbot, question_input])
    clear_button.click(lambda: initial_history, outputs=chatbot)

iface.launch()
