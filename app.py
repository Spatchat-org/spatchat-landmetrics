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
    "pland":          ("Proportion of Landscape (PLAND)",      "Percentage of landscape comprised by the class."),
    "np":             ("Number of Patches (NP)",               "Total number of patches for the class or entire landscape."),
    "pd":             ("Patch Density (PD)",                   "Number of patches per 100 hectares."),
    "lpi":            ("Largest Patch Index (LPI)",            "Percentage of total landscape made up by the largest patch."),
    "te":             ("Total Edge (TE)",                      "Total length of all patch edges."),
    "edge_density":   ("Edge Density (ED)",                    "Edge length per hectare."),
    "lsi":            ("Landscape Shape Index (LSI)",          "Overall shape complexity of the landscape."),
    "tca":            ("Total Core Area (TCA)",                "Sum of all core areas in the landscape."),
    "mesh":           ("Effective Mesh Size (MESH)",           "Average patch size accounting for edges."),
    "contag":         ("Contagion Index (CONTAG)",             "Clumpiness of patches."),
    "shdi":           ("Shannon Diversity Index (SHDI)",       "Diversity of patch types."),
    "shei":           ("Shannon Evenness Index (SHEI)",        "Evenness of patch distribution."),
    "area":           ("Total Area (AREA)",                    "Total area of patches by class or landscape."),
    "perim":          ("Total Perimeter (PERIM)",               "Total perimeter of all patches."),
    "para":           ("Perimeter-Area Ratio (PARA)",          "Ratio of perimeter to area."),
    "shape":          ("Shape Index (SHAPE)",                  "Normalized perimeter-area ratio."),
    "frac":           ("Fractal Dimension (FRAC)",             "Fractal complexity of patch shapes."),
    "enn":            ("Euclidean Nearest Neighbor (ENN)",     "Distance to nearest patch of same class."),
    "core":           ("Total Core Area (CORE)",               "Sum of interior (nondisturbed) patch area."),
    "nca":            ("Number of Core Areas (NCA)",           "Count of core regions in the landscape."),
    "cai":            ("Core Area Index (CAI)",                "Proportion of core area to total patch area."),
}

# --- Synonyms for natural names ---
synonyms = {
    "edge_density": ["ed", "edge density", "edge_density"],
    "pland":          ["pland", "proportion of landscape"],
    "np":             ["np", "number of patches"],
    "pd":             ["pd", "patch density"],
    "lpi":            ["lpi", "largest patch"],
    "te":             ["te", "total edge", "total_edge"],
    # add more as desired...
}

# --- Map our codes to actual DataFrame column names ---
col_map = {
    "pland":         "proportion_of_landscape",
    "np":            "number_of_patches",
    "pd":            "patch_density",
    "lpi":           "largest_patch_index",
    "te":            "total_edge",
    "edge_density":  "edge_density",
    "lsi":           "landscape_shape_index",
    "tca":           "total_core_area",
    "mesh":          "effective_mesh_size",
    "contag":        "contagion_index",
    "shdi":          "shannon_diversity_index",
    "shei":          "shannon_evenness_index",
    "area":          "total_area",
    "perim":         "total_perimeter",
    "para":          "perimeter_area_ratio",
    "shape":         "shape_index",
    "frac":          "fractal_dimension_index",
    "enn":           "euclidean_nearest_neighbor_distance",
    "core":          "total_core_area",
    "nca":           "number_of_core_areas",
    "cai":           "core_area_index",
}

# --- Raster preview & clear ---

def no_raster_fig():
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.text(0.5, 0.5, "🗂️ No raster loaded.", ha='center', va='center', color='gray', fontsize=14)
    ax.set_title("Raster Preview", color='dimgray')
    ax.axis('off')
    return fig

def preview_raster(file, mapping_file):
    try:
        # 0) If nothing uploaded yet
        if file is None:
            return no_raster_fig()

        # 1) Load optional CSV mapping into code2name dict
        code2name = {}
        if mapping_file is not None:
            df_map = pd.read_csv(mapping_file.name)
            code2name = {int(r['code']): r['name'] for _, r in df_map.iterrows()}

        # 2) Read the raster band + categories
        with rasterio.open(file.name) as src:
            raw    = src.read(1)
            nodata = src.nodata or 0
            try:
                cats = src.categories(1) or {}
            except:
                cats = {}

        # 3) Decide which labeling branch to use
        if code2name:
            data   = raw.astype(int)
            vals   = sorted(code2name.keys())
            labels = [f"{v}: {code2name[v]}" for v in vals]
            n      = len(vals)
            colors = plt.cm.tab10(np.linspace(0, 1, n))

        elif cats:
            data   = raw.astype(int)
            vals   = sorted(cats.keys())
            labels = [f"{v}: {cats[v]}" for v in vals]
            n      = len(vals)
            colors = plt.cm.tab10(np.linspace(0, 1, n))

        elif raw.dtype.kind in ('U','S','O'):
            data_str     = raw.astype(str)
            unique_names = np.unique(data_str[data_str != ""])
            name2code    = {nm: i+1 for i, nm in enumerate(unique_names)}
            data         = np.zeros_like(raw, dtype=int)
            for nm, code in name2code.items():
                data[data_str == nm] = code
            n      = len(unique_names)
            labels = [f"{i+1}: {unique_names[i]}" for i in range(n)]
            colors = plt.cm.tab10(np.linspace(0, 1, n))

        else:
            data   = raw
            vals   = np.unique(data[data != nodata])
            labels = [f"{int(v)}: Class {int(v)}" for v in vals]
            n      = len(vals)
            colors = plt.cm.tab10(np.linspace(0, 1, n))

        # 4) Plot
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(data, cmap='tab10', interpolation='nearest', vmin=1, vmax=n)
        ax.set_title("Uploaded Raster")
        ax.axis('off')

        # 5) Legend
        handles = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(n)]
        ax.legend(handles=handles, loc='lower left', fontsize='small', frameon=True)

        return fig

    except Exception as e:
        print("preview_raster error:", e)
        return no_raster_fig()

def clear_raster():
    fig, ax = plt.subplots(figsize=(5,5))
    ax.text(0.5,0.5,"🗂️ No raster loaded.", ha='center', va='center', color='gray')
    ax.set_title("Raster Preview", color='dimgray')
    ax.axis('off')
    return None, gr.update(value=fig, visible=True)

# --- Handlers ---
def answer_metadata(file, history):
    with rasterio.open(file.name) as src:
        crs     = src.crs
        x_res,y_res = src.res
        extent  = src.bounds
        bands   = src.count
        nodata  = src.nodata
        unit    = getattr(src.crs,"linear_units","unit")
    text = "\n".join([
        f"CRS: {crs}",
        f"Resolution: {x_res:.2f} × {y_res:.2f} {unit}",
        f"Extent: {extent}",
        f"Bands: {bands}",
        f"NoData value: {nodata}"
    ])
    return history + [{"role":"assistant","content":text}], ""

def list_metrics(history):
    groups = {
        "Landscape": ["contag","shdi","shei","mesh","lsi","tca"],
        "Class":     ["pland","np","pd","lpi","te","edge_density"],
        "Patch":     ["area","perim","para","shape","frac","enn","core","nca","cai"]
    }
    lines = []
    for lvl, keys in groups.items():
        lines.append(f"**{lvl}-level metrics:**")
        for k in keys:
            name,_ = metric_definitions[k]
            lines.append(f"- {name} (`{k}`)")
        lines.append("")
    return history + [{"role":"assistant","content":"\n".join(lines).strip()}], ""

def compute_metric(file, key, history, mapping_file):
    # 1) Load user’s code→name CSV if provided
    code2name = {}
    if mapping_file is not None:
        try:
            df_map = pd.read_csv(mapping_file.name)
            # expect columns: code,name
            code2name = {int(r['code']): str(r['name']) for _, r in df_map.iterrows()}
        except Exception as e:
            print("mapping load error:", e)

    # 2) Read raster + resolution
    with rasterio.open(file.name) as src:
        raw    = src.read(1)
        x_res, y_res = src.res
        nodata = src.nodata or 0

    # 3) Remap string‐typed categories → ints if needed
    if raw.dtype.kind in ('U','S','O'):
        data_str    = raw.astype(str)
        unique_names = np.unique(data_str[data_str != ""])
        name2code   = {nm: i+1 for i, nm in enumerate(unique_names)}
        arr = np.zeros_like(raw, dtype=int)
        for nm, code in name2code.items():
            arr[data_str == nm] = code
        # override any existing code2name
        code2name = {v: k for k, v in name2code.items()}
        ls = Landscape(arr, res=(x_res, y_res), nodata=0)
    else:
        ls = Landscape(file.name, nodata=nodata, res=(x_res, y_res))

    # 4) Compute metrics DataFrames
    df_land = ls.compute_landscape_metrics_df()
    df_cls  = ls.compute_class_metrics_df()

    # 5) Get human‐readable metric name & df column
    metric_name, _ = metric_definitions[key]
    col = col_map.get(key, key)

    # 6) Build landscape‐level text
    if key == "np":
        val = int(df_cls["number_of_patches"].sum())
        land_part = f"**Landscape-level {metric_name}:** {val}\n\n"
    elif col in df_land.columns:
        val = df_land[col].iloc[0]
        land_part = f"**Landscape-level {metric_name}:** {val:.4f}\n\n"
    else:
        land_part = ""

    # 7) Build class‐level table
    df2 = df_cls.rename_axis("code").reset_index()
    # label each row as "code: name" if mapping exists
    if code2name:
        df2["class_name"] = df2["code"].map(
            lambda c: f"{c}: {code2name.get(c, 'Unknown')}"
        )
    else:
        df2["class_name"] = df2["code"].map(lambda c: f"Class {int(c)}")

    if col in df2.columns:
        tbl = df2[["class_name", col]].to_markdown(index=False)
    else:
        tbl = "(not available at class level)"

    # 8) Compose response
    content = land_part + f"**Class-level {metric_name}:**\n{tbl}"
    return history + [{"role":"assistant","content":content}], ""

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
            "You are Spatchat, a helpful assistant. "
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

# --- Main handler ---
def analyze_raster(file, mapping_file, question, history):
    hist  = history + [{"role":"user","content":question}]
    lower = question.lower()

    # require raster
    if file is None:
        return hist + [{"role":"assistant","content":
            "Please upload a GeoTIFF before asking anything."
        }], ""

    # how many classes?
    if "how many classes" in lower or re.search(r"\bnum(ber)? of classes\b", lower):
        return count_classes(file, hist)

    # metadata?
    if re.search(r"\b(crs|resolution|extent|bands|nodata)\b", lower):
        return answer_metadata(file, hist)

    # list metrics?
    if re.search(r"\b(what|which|list|available).*metrics\b", lower):
        return list_metrics(hist)

    # compute metric?
    for code,(full,_) in metric_definitions.items():
        for syn in synonyms.get(code, [code, full.lower()]):
            if re.search(rf"\b{re.escape(syn)}\b", lower):
                return compute_metric(file, code, hist)

    # fallback
    return llm_fallback(hist)

# --- UI layout ---
initial_history = [
    {"role":"assistant","content":
     "👋 Hi! I’m Spatchat. Upload a GeoTIFF to begin—then ask for CRS, resolution, extent, or any landscape metric."}
]

# --- UI layout ---
with gr.Blocks(title="Spatchat") as iface:
    # logo & share link
    gr.HTML('<head><link rel="icon" href="file=logo1.png"></head>')
    gr.Image(
        value="logo/logo_long1.png", type="filepath",
        show_label=False, show_download_button=False,
        show_share_button=False, elem_id="logo-img"
    )
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
            mapping_input       = gr.File(label="Upload code→name CSV", file_types=['csv'])
            raster_output       = gr.Plot(label="Raster Preview")
            clear_raster_button = gr.Button("Clear Raster")
        with gr.Column(scale=1):
            chatbot        = gr.Chatbot(value=initial_history, type="messages",
                                       label="Spatchat Dialog")
            question_input = gr.Textbox(placeholder="e.g. calculate edge density",
                                       lines=1)
            clear_button   = gr.Button("Clear Chat")

    # callbacks
    file_input.change(preview_raster,
                      inputs=[file_input, mapping_input],
                      outputs=raster_output)
    mapping_input.change(preview_raster,
                         inputs=[file_input, mapping_input],
                         outputs=raster_output)
    clear_raster_button.click(clear_raster,
                              inputs=None,
                              outputs=[file_input, raster_output])
    question_input.submit(analyze_raster,
                          inputs=[file_input, mapping_input, question_input, chatbot],
                          outputs=[chatbot, question_input])
    clear_button.click(lambda: initial_history, outputs=chatbot)

iface.launch()

