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

# --- Hard‑coded metric glossary ---
metric_definitions = {
    "pland":  ("Proportion of Landscape (PLAND)",      "Percentage of landscape comprised by the class."),
    "np":     ("Number of Patches (NP)",               "Total number of patches for the class or landscape."),
    "pd":     ("Patch Density (PD)",                   "Number of patches per 100 hectares."),
    "lpi":    ("Largest Patch Index (LPI)",            "Percentage of total landscape made up by the largest patch."),
    "te":     ("Total Edge (TE)",                      "Total length of all patch edges."),
    "edge_density": ("Edge Density (ED)",             "Edge length per hectare."),
    "lsi":    ("Landscape Shape Index (LSI)",          "Overall shape complexity of the landscape."),
    "tca":    ("Total Core Area (TCA)",                "Sum of all core areas in the landscape."),
    "mesh":   ("Effective Mesh Size (MESH)",           "Average size of patches after accounting for fragmentation."),
    "contag": ("Contagion Index (CONTAG)",             "Clumpiness of patches — higher means more aggregated."),
    "shdi":   ("Shannon Diversity Index (SHDI)",       "Diversity of patch types."),
    "shei":   ("Shannon Evenness Index (SHEI)",        "Evenness of patch distribution."),
    "area":   ("Patch Area (AREA)",                    "Area of each individual patch."),
    "perim":  ("Patch Perimeter (PERIM)",              "Perimeter of each patch."),
    "para":   ("Perimeter-Area Ratio (PARA)",          "Ratio of perimeter to area for each patch."),
    "shape":  ("Shape Index (SHAPE)",                  "Shape complexity of each patch."),
    "frac":   ("Fractal Dimension (FRAC)",             "Fractal complexity of each patch shape."),
    "enn":    ("Euclidean Nearest Neighbor (ENN)",     "Distance to nearest patch of same class."),
    "core":   ("Core Area (CORE)",                     "Interior area of a patch excluding edges."),
    "nca":    ("Number of Core Areas (NCA)",           "Total core regions in the landscape."),
    "cai":    ("Core Area Index (CAI)",                "Proportion of core area to total patch area."),
}

# --- Synonyms for natural language matching ---
synonyms = {
    "edge_density": ["ed", "edge density", "edge_density"],
    "pland":          ["pland", "proportion of landscape", "percent landscape"],
    "np":             ["np", "number of patches"],
    "pd":             ["pd", "patch density"],
    # … add more if needed …
}

# --- Raster preview & clear ---
def preview_raster(file):
    try:
        with rasterio.open(file.name) as src:
            data = src.read(1)
            nodata = src.nodata
        unique = np.unique(data[data != nodata])
        colors = plt.cm.tab10(np.linspace(0,1,len(unique)))
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(data, cmap='tab10', interpolation='nearest')
        ax.set_title("Uploaded Raster")
        ax.axis('off')
        handles = [
            mpatches.Patch(color=colors[i], label=f"Class {int(val)}")
            for i,val in enumerate(unique)
        ]
        ax.legend(handles=handles, loc='lower left', fontsize='small', frameon=True)
        return fig
    except:
        fig, ax = plt.subplots(figsize=(5,5))
        ax.text(0.5,0.5,"🗂️ No raster loaded.", ha='center', va='center', color='gray')
        ax.set_title("Raster Preview", color='dimgray')
        ax.axis('off')
        return fig

def clear_raster():
    fig, ax = plt.subplots(figsize=(5,5))
    ax.text(0.5,0.5,"🗂️ No raster loaded.", ha='center', va='center', color='gray')
    ax.set_title("Raster Preview", color='dimgray')
    ax.axis('off')
    return None, gr.update(value=fig, visible=True)

# --- Handlers ---
def answer_metadata(file, history):
    with rasterio.open(file.name) as src:
        crs      = src.crs
        x_res,y_res = src.res
        extent   = src.bounds
        bands    = src.count
        nodata   = src.nodata
        unit     = getattr(src.crs,"linear_units","unit")
    txt = "\n".join([
        f"CRS: {crs}",
        f"Resolution: {x_res:.2f} × {y_res:.2f} {unit}",
        f"Extent: {extent}",
        f"Bands: {bands}",
        f"NoData value: {nodata}"
    ])
    return history + [{"role":"assistant","content":txt}], ""

def list_metrics(history):
    groups = {
        "Landscape":   ["contag","shdi","shei","mesh","lsi","tca"],
        "Class":       ["pland","np","pd","lpi","te","edge_density"],
        "Patch":       ["area","perim","para","shape","frac","enn","core","nca","cai"]
    }
    lines = []
    for lvl, keys in groups.items():
        lines.append(f"**{lvl}-level metrics:**")
        for k in keys:
            name,_ = metric_definitions[k]
            lines.append(f"- {name} (`{k}`)")
        lines.append("")
    return history + [{"role":"assistant","content":"\n".join(lines).strip()}], ""

def compute_metric(file, key, history):
    ls       = Landscape(file.name, nodata=0)
    df_land  = ls.compute_landscape_metrics_df()
    df_class = ls.compute_class_metrics_df()
    name,_   = metric_definitions[key]

    land_part = ""
    if key in df_land.columns:
        val = df_land[key].iloc[0]
        land_part = f"**Landscape-level {name}:** {val:.4f}\n\n"

    df2 = df_class.rename_axis("code").reset_index()
    df2["class_name"] = df2["code"].map(lambda c: f"Class {int(c)}")
    tbl = df2[["class_name", key]].to_markdown(index=False)

    content = land_part + f"**Class-level {name}:**\n{tbl}"
    return history + [{"role":"assistant","content":content}], ""

def llm_fallback(history):
    prompt = [
        {"role":"system","content":(
            "You are Spatchat. Use rasterio for metadata and pylandstats for metrics; otherwise reply conversationally."
        )},
        *history
    ]
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=prompt,
        temperature=0.4
    ).choices[0].message.content
    return history + [{"role":"assistant","content":resp}], ""

# --- Main analyzer with regex intent detection ---
def analyze_raster(file, question, history):
    hist = history + [{"role":"user","content":question}]
    lower = question.lower()

    # 1) require file before anything else
    if file is None:
        return hist + [{"role":"assistant",
            "content":"Please upload a GeoTIFF before asking anything."}], ""

    # 2) metadata?
    if re.search(r"\b(crs|resolution|extent|bands|nodata)\b", lower):
        return answer_metadata(file, hist)

    # 3) list metrics?
    if re.search(r"\b(what|which|list|available).*metrics\b", lower):
        return list_metrics(hist)

    # 4) compute metric?
    for code, (full, _) in metric_definitions.items():
        for syn in synonyms.get(code, [code, full.lower()]):
            if re.search(rf"\b{re.escape(syn)}\b", lower):
                return compute_metric(file, code, hist)

    # 5) fallback
    return llm_fallback(hist)

# --- UI layout ---
initial_history = [
    {"role":"assistant","content":
     "👋 Hi! I’m Spatchat. Upload a GeoTIFF to begin—then ask for CRS, resolution, extent, or any landscape metric."}
]

with gr.Blocks(title="Spatchat") as iface:
    # logo & share link
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

    # callbacks
    file_input.change(preview_raster, inputs=file_input, outputs=raster_output)
    clear_raster_button.click(clear_raster, inputs=None,
                              outputs=[file_input, raster_output])
    question_input.submit(analyze_raster,
                          inputs=[file_input, question_input, chatbot],
                          outputs=[chatbot, question_input])
    clear_button.click(lambda: initial_history, outputs=chatbot)

iface.launch()
