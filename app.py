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

# --- Map our codes to actual DataFrame column names ---
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

# --- Helper mappings for metrics ---
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

# --- Raster preview & clear ---
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
            labels = [f"{int(v)}: Class {int(v)}" for v in vals]
            n = len(vals)

        colors = plt.cm.tab10(np.linspace(0,1,n))
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(data, cmap='tab10', interpolation='nearest', vmin=1, vmax=n)
        ax.set_title("Uploaded Raster")
        ax.axis('off')
        handles = [mpatches.Patch(color=colors[i], label=labels[i])
                   for i in range(n)]
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

# --- Metrics listing ---
def list_metrics(history):
    lines = []
    lines.append("**Cross‑level metrics (Landscape & Class):**")
    for k in cross_level:
        name, _ = metric_definitions[k]
        lines.append(f"- {name} (`{k}`)")
    lines.append("")
    lines.append("**Landscape‑only metrics:**")
    for k in helper_methods:
        name, _ = metric_definitions[k]
        lines.append(f"- {name} (`{k}`)")
    lines.append("")
    lines.append("**Class‑only metrics:**")
    for k in class_only:
        name, _ = metric_definitions[k]
        lines.append(f"- {name} (`{k}`)")
    content = "\n".join(lines)
    return history + [{"role": "assistant", "content": content}], ""

# --- Core compute helpers ---
def _build_landscape(file):
    with rasterio.open(file.name) as src:
        raw = src.read(1)
        x_res, y_res = src.res
        nodata = src.nodata or 0
    if raw.dtype.kind in ("U", "S", "O"):
        data_str = raw.astype(str)
        uniq = np.unique(data_str[data_str!=""])
        name2code = {nm: i+1 for i, nm in enumerate(uniq)}
        arr = np.zeros_like(raw, dtype=int)
        for nm, code in name2code.items():
            arr[data_str==nm] = code
        ls = Landscape(arr, res=(x_res, y_res), nodata=0)
    else:
        ls = Landscape(file.name, nodata=nodata, res=(x_res, y_res))
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
    return history + [{"role": "assistant", "content": content}], ""


def compute_class_only(file, keys, history):
    ls = _build_landscape(file)
    cols = [col_map[k] for k in keys]
    df = ls.compute_class_metrics_df(metrics=cols).rename_axis("code").reset_index()
    df["class_name"] = df["code"].map(lambda c: f"Class {int(c)}")
    tbl = df[["class_name"] + cols].to_markdown(index=False)
    content = f"**Class-level metrics:**\n{tbl}"
    return history + [{"role": "assistant", "content": content}], ""


def compute_multiple_metrics(file, keys, history):
    # split into what we can do at landscape vs. class levels
    land_keys  = [k for k in keys if k not in class_only]
    class_keys = keys
    hl, _ = compute_landscape_only(file, land_keys, history)
    hc, _ = compute_class_only(file, class_keys, history)
    land_msg = hl[-1]
    class_msg = hc[-1]
    return history + [land_msg, class_msg], ""
    
# --- Main handler ---
def analyze_raster(file, question, history):
    hist  = history + [{"role": "user", "content": question}]
    lower = question.lower()

    # 1️⃣ Always allow "list metrics" without a file
    if re.search(r"\b(list|available).*metrics\b", lower):
        return list_metrics(hist)

    # 2️⃣ Other simple handlers
    if file is None:
        return hist + [{"role": "assistant", "content": "Please upload a GeoTIFF before asking anything."}], ""
    if "how many classes" in lower or re.search(r"\bnum(ber)? of classes\b", lower):
        return count_classes(file, hist)
    if re.search(r"\b(crs|resolution|extent|bands|nodata)\b", lower):
        return answer_metadata(file, hist)

    # 3️⃣ Detect explicit level qualifiers
    is_land  = bool(re.search(r"\blandscape[- ]level\b", lower))
    is_class = bool(re.search(r"\bclass[- ]level\b",     lower))

    # 4️⃣ Handle "all ... metrics" requests first
    if is_land and "all" in lower:
        # all metrics except strictly class-only
        found = [c for c in metric_definitions if c not in class_only]
    elif is_class and "all" in lower:
        found = list(class_only)
    else:
        # 5️⃣ Otherwise, pick out only those keys mentioned
        found = []
        for code, (fullname, _) in metric_definitions.items():
            for syn in synonyms.get(code, [code, fullname.lower()]):
                if re.search(rf"\b{re.escape(syn)}\b", lower) and code not in found:
                    found.append(code)

    # 6️⃣ If nothing matched, send to LLM fallback
    if not found:
        return llm_fallback(hist)

    # 7️⃣ Branch on multi vs. single
    if len(found) > 1:
        if is_land:
            return compute_landscape_only(file, found, hist)
        if is_class:
            return compute_class_only(file, found, hist)
        return compute_multiple_metrics(file, found, hist)
    else:
        key = found[0]
        if is_land:
            return compute_landscape_only(file, [key], hist)
        if is_class:
            return compute_class_only(file, [key], hist)
        # default: show both levels for that single metric
        return compute_multiple_metrics(file, [key], hist)


def count_classes(file, history):
    with rasterio.open(file.name) as src:
        raw = src.read(1)
        nodata = src.nodata or 0
    if raw.dtype.kind in ('U','S','O'):
        vals = np.unique(raw.astype(str)[raw.astype(str)!=""])
    else:
        vals = np.unique(raw[raw!=nodata])
    return history + [{
        "role":"assistant",
        "content":f"Your raster contains {len(vals)} unique classes."
    }], ""

def llm_fallback(history):
    prompt = [
        {"role":"system","content":(
            "You are Spatchat, a helpful assistant in calculating and summarizing landscape metrics. \n"
            "Use rasterio for metadata and pylandstats for metrics; otherwise be conversational."
        )},
        *history
    ]
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=prompt,
        temperature=0.4
    ).choices[0].message.content
    return history + [{"role":"assistant","content":resp}], ""

# --- UI layout ---
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
    ''')

    with gr.Row():
        with gr.Column(scale=1):
            file_input          = gr.File(label="Upload GeoTIFF", type="filepath")
            raster_output       = gr.Plot(label="Raster Preview")
            clear_raster_button = gr.Button("Clear Raster")
        with gr.Column(scale=1):
            chatbot        = gr.Chatbot(value=initial_history, type="messages",
                                       label="Spatchat Dialog")
            question_input = gr.Textbox(placeholder="e.g. calculate edge density",
                                       lines=1)
            clear_button   = gr.Button("Clear Chat")

    file_input.change(preview_raster, inputs=file_input, outputs=raster_output)
    clear_raster_button.click(clear_raster, inputs=None,
                              outputs=[file_input, raster_output])
    question_input.submit(analyze_raster,
                          inputs=[file_input, question_input, chatbot],
                          outputs=[chatbot, question_input])
    clear_button.click(lambda: initial_history, outputs=chatbot)

iface.launch()
