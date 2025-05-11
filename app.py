import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
import numpy as np
import re
from pylandstats import Landscape
from together import Together
import os
from dotenv import load_dotenv

# --- LLM setup ---
load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# --- Metric glossary (hard-coded) ---
metric_definitions = {
    "pland":  ("Proportion of Landscape (PLAND)",      "% of landscape by class"),
    "np":     ("Number of Patches (NP)",               "count of patches"),
    "pd":     ("Patch Density (PD)",                   "patches per 100 ha"),
    "lpi":    ("Largest Patch Index (LPI)",            "% of landscape by largest patch"),
    "te":     ("Total Edge (TE)",                      "total edge length"),
    "edge_density": ("Edge Density (ED)",             "edge length per ha"),
    "lsi":    ("Landscape Shape Index (LSI)",          "shape complexity"),
    "tca":    ("Total Core Area (TCA)",                "sum of core areas"),
    "mesh":   ("Effective Mesh Size (MESH)",           "avg patch size accounting edges"),
    "contag": ("Contagion Index (CONTAG)",             "aggregation/clumping measure"),
    "shdi":   ("Shannon Diversity Index (SHDI)",       "diversity of patch types"),
    "shei":   ("Shannon Evenness Index (SHEI)",        "evenness of patch types"),
    "area":   ("Patch Area (AREA)",                    "area of individual patches"),
    "perim":  ("Patch Perimeter (PERIM)",              "perimeter of patches"),
    "para":   ("Perimeter-Area Ratio (PARA)",          "perimeter/area ratio"),
    "shape":  ("Shape Index (SHAPE)",                  "shape complexity per patch"),
    "frac":   ("Fractal Dimension (FRAC)",             "fractal complexity"),
    "enn":    ("Euclidean Nearest Neighbor (ENN)",     "nearest neighbor distance"),
    "core":   ("Core Area (CORE)",                     "interior area excluding edge"),
    "nca":    ("Number of Core Areas (NCA)",           "count of core areas"),
    "cai":    ("Core Area Index (CAI)",                "core area proportion"),
    "total_edge": ("Total Edge (TE)",                  "total edge length"),
}

# --- Preview & clear raster ---
def preview_raster(file):
    try:
        with rasterio.open(file.name) as src:
            data = src.read(1)
            nodata = src.nodata
        unique = np.unique(data[data != nodata])
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique)))
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
        ax.text(0.5,0.5,"🗂️ No raster loaded.",ha='center',va='center',color='gray')
        ax.set_title("Raster Preview",color='dimgray')
        ax.axis('off')
        return fig

def clear_raster():
    fig, ax = plt.subplots(figsize=(5,5))
    ax.text(0.5,0.5,"🗂️ No raster loaded.",ha='center',va='center',color='gray')
    ax.set_title("Raster Preview",color='dimgray')
    ax.axis('off')
    return None, gr.update(value=fig, visible=True)

# --- Intent branches ---
def answer_metadata(file, history):
    with rasterio.open(file.name) as src:
        crs = src.crs
        x_res,y_res = src.res
        extent = src.bounds
        bands = src.count
        nodata = src.nodata
        unit = getattr(src.crs,"linear_units","unit")
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
        "Class":     ["pland","np","pd","lpi","total_edge","edge_density"],
        "Patch":     ["area","perim","para","shape","frac","enn","core","nca","cai"]
    }
    lines=[]
    for lvl, keys in groups.items():
        lines.append(f"**{lvl}-level metrics:**")
        for k in keys:
            name,_ = metric_definitions[k]
            lines.append(f"- {name} (`{k}`)")
        lines.append("")
    return history + [{"role":"assistant","content":"\n".join(lines).strip()}], ""

def compute_metric(file, key, history):
    ls = Landscape(file.name, nodata=0)
    df_land = ls.compute_landscape_metrics_df()
    df_cls  = ls.compute_class_metrics_df()
    name,_  = metric_definitions[key]

    land_msg=""
    if key in df_land.columns:
        val = df_land[key].iloc[0]
        land_msg = f"**Landscape-level {name}:** {val:.4f}\n\n"

    df2 = df_cls.rename_axis("code").reset_index()
    df2["class_name"] = df2["code"].map(lambda c: f"Class {int(c)}")
    tbl = df2[["class_name", key]].to_markdown(index=False)
    content = land_msg + f"**Class-level {name}:**\n{tbl}"
    return history + [{"role":"assistant","content":content}], ""

# --- Fallback to LLM ---
def llm_fallback(history):
    prompt = [
        {"role":"system","content":(
            "You are Spatchat, a friendly assistant. Use rasterio for metadata "
            "and pylandstats for metrics when asked; otherwise reply conversationally."
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
def analyze_raster(file, question, history):
    hist = history + [{"role":"user","content":question}]
    lower = question.lower()

    # 1) require file
    if file is None:
        return hist + [{"role":"assistant",
            "content":"Please upload a GeoTIFF before asking anything."
        }], ""

    # 2) metadata?
    if re.search(r"\b(crs|resolution|extent|bands|nodata)\b", lower):
        return answer_metadata(file, hist)

    # 3) list metrics?
    if re.search(r"\b(what|which|list|available).*metrics\b", lower):
        return list_metrics(hist)

    # 4) compute metric?
    #    look for any code or spaced name
    for code in metric_definitions:
        name,_ = metric_definitions[code]
        if re.search(rf"\b{re.escape(code)}\b", lower) \
        or re.search(rf"\b{re.escape(name.lower())}\b", lower):
            return compute_metric(file, code, hist)

    # 5) fallback
    return llm_fallback(hist)

# --- UI layout ---
initial_history = [
    {"role":"assistant","content":
     "👋 Hi! I’m Spatchat. Upload a GeoTIFF to begin—then ask for CRS, resolution, or any landscape metric."}
]

with gr.Blocks(title="Spatchat") as iface:
    # logo & share
    gr.HTML('<link rel="icon" href="file=logo1.png">')
    gr.Image("logo/logo_long1.png", type="filepath", show_label=False,
             show_download_button=False, show_share_button=False,
             elem_id="logo-img")
    gr.HTML("<style>#logo-img img{height:90px;margin:10px;border-radius:6px;}</style>")
    gr.Markdown("## 🌲 Spatchat: Landscape Metrics Assistant")
    gr.HTML('''
      <div style="margin:-10px 0 20px;">
        <input id="shareLink" readonly value="https://spatchat.org/browse/?room=landmetrics"
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

    # bindings
    file_input.change(preview_raster, inputs=file_input, outputs=raster_output)
    clear_raster_button.click(clear_raster, inputs=None,
                              outputs=[file_input, raster_output])
    question_input.submit(analyze_raster,
                          inputs=[file_input, question_input, chatbot],
                          outputs=[chatbot, question_input])
    clear_button.click(lambda: initial_history, outputs=chatbot)

iface.launch()
