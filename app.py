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
    "shdi":         ("Shannon Diversity Index (SHDI)", "Diversity of types"),
    "shei":         ("Shannon Evenness Index (SHEI)", "Evenness of distribution"),
    "area":         ("Total Area (AREA)", "Area by patch"),
    "perim":        ("Total Perimeter (PERIM)", "Perimeter of patches"),
    "para":         ("Perimeter-Area Ratio (PARA)", "Perimeter/area"),
    "shape":        ("Shape Index (SHAPE)", "Normalized shape"),
    "frac":         ("Fractal Dimension (FRAC)", "Shape complexity"),
    "enn":          ("Euclidean Nearest Neighbor (ENN)", "Distance to nearest patch"),
    "core":         ("Total Core Area (CORE)", "Interior core area"),
    "nca":          ("Number of Core Areas (NCA)", "Count core regions"),
    "cai":          ("Core Area Index (CAI)", "% core area to patch area"),
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

# --- Column map ---
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
    "contag":       "contagion",
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
    "pd":"patch_density",
    "edge_density":"edge_density",
    "lsi":"landscape_shape_index",
    "contag":"contagion",
    "shdi":"shannon_diversity_index",
    "shei":"shannon_evenness_index",
    "mesh":"effective_mesh_size",
    "tca":"total_core_area",
    "lpi":"largest_patch_index",
    "te":"total_edge",
}
class_only = {"pland","area","perim","para","shape","frac","enn","core","nca","cai"}
cross_level = ["np","pd","lpi","te","edge_density"]

# --- Raster preview and notification ---
def no_raster_fig():
    fig, ax = plt.subplots(figsize=(5,5))
    ax.text(0.5,0.5,"🗂️ No raster loaded.",ha='center',va='center',color='gray')
    ax.set_title("Raster Preview",color='dimgray')
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
            data = raw.astype(str)
            unique = np.unique(data[data!=""])
            arr = np.zeros_like(raw,dtype=int)
            for i,nm in enumerate(unique,1): arr[data==nm] = i
            labels = [f"{i}: {nm}" for i,nm in enumerate(unique,1)]
        else:
            arr = raw
            vals = np.unique(arr[arr!=nodata])
            labels = [f"Class {int(v)}" for v in vals]
        n = len(labels)
        cmap = plt.cm.tab10(np.linspace(0,1,n))
        fig,ax = plt.subplots(figsize=(5,5))
        ax.imshow(arr,cmap='tab10')
        ax.axis('off')
        ax.set_title("Uploaded Raster")
        handles = [mpatches.Patch(color=c,label=l) for c,l in zip(cmap,labels)]
        ax.legend(handles=handles,loc='lower left',fontsize='small')
        return fig
    except:
        return no_raster_fig()

def notify_upload(file,history):
    if file:
        return history + [{"role":"assistant","content":"📥 Raster uploaded successfully!"}], ""
    return history, ""

def clear_raster():
    return None, gr.update(value=no_raster_fig(),visible=True)

# --- Handlers ---
def answer_metadata(file,history):
    with rasterio.open(file.name) as src:
        crs = src.crs
        x_res,y_res = src.res
        extent = src.bounds
        bands = src.count
        nodata = src.nodata
        unit = getattr(src.crs,'linear_units','unit')
    text = f"CRS: {crs}\nResolution: {x_res:.2f}×{y_res:.2f} {unit}\nExtent: {extent}\nBands: {bands}\nNoData: {nodata}"
    return history + [{"role":"assistant","content":text}], ""

def count_classes(file,history):
    with rasterio.open(file.name) as src:
        arr = src.read(1)
        nodata = src.nodata or 0
    vals = np.unique(arr[arr!=nodata])
    return history + [{"role":"assistant","content":f"Your raster contains {len(vals)} unique classes."}], ""

def list_metrics(history):
    lines=["**Cross‑level metrics:**"]
    lines += [f"- {metric_definitions[k][0]} (`{k}`)" for k in cross_level]
    lines.append("\n**Landscape‑only metrics:**")
    lines += [f"- {metric_definitions[k][0]} (`{k}`)" for k in helper_methods]
    lines.append("\n**Class‑only metrics:**")
    lines += [f"- {metric_definitions[k][0]} (`{k}`)" for k in class_only]
    return history + [{"role":"assistant","content":"\n".join(lines)}], ""

def compute_landscape_only(file,keys,history):
    ls = Landscape(file.name, nodata=getattr(rasterio.open(file.name),'nodata',0), res=tuple(rasterio.open(file.name).res))
    parts=[]
    for key in keys:
        name,_ = metric_definitions[key]
        if key=="np":
            df = ls.compute_class_metrics_df(metrics=["number_of_patches"])
            val = int(df["number_of_patches"].sum())
        elif key in helper_methods:
            val = getattr(ls, helper_methods[key])()
        else:
            df_l = ls.compute_landscape_metrics_df(metrics=[col_map[key]])
            val = df_l[col_map[key]].iloc[0]
        parts.append(f"**{name} ({key.upper()}):** {val:.4f}" if isinstance(val,float) else f"**{name} ({key.upper()}):** {val}")
    return history + [{"role":"assistant","content":"\n\n".join(parts)}], ""

def compute_class_only(file,keys,history):
    ls = Landscape(file.name, nodata=getattr(rasterio.open(file.name),'nodata',0), res=tuple(rasterio.open(file.name).res))
    cols = [col_map[k] for k in keys]
    df = ls.compute_class_metrics_df(metrics=cols).rename_axis("code").reset_index()
    df["class_name"] = df["code"].map(lambda c: f"Class {int(c)}")
    tbl = df[["class_name"]+cols].to_markdown(index=False)
    return history + [{"role":"assistant","content":f"**Class-level metrics:**\n{tbl}"}], ""

def compute_multiple_metrics(file,keys,history):
    hl,_ = compute_landscape_only(file,keys,history)
    hc,_ = compute_class_only(file,keys,history)
    return history + [hl[-1], hc[-1]], ""

def llm_fallback(history):
    prompt=[{"role":"system","content":"You are Spatchat: use rasterio & pylandstats; otherwise be conversational."}, *history]
    resp = client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free", messages=prompt, temperature=0.4).choices[0].message.content
    return history + [{"role":"assistant","content":resp}], ""

# --- Main handler ---
def analyze_raster(file,question,history):
    history,_ = notify_upload(file,history)
    hist = history + [{"role":"user","content":question}]
    lower = question.lower()
    parsed = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=[
            {"role":"system","content":"Parse JSON: list_metrics,count_classes,metadata,metrics,level,all_metrics."},
            {"role":"user","content":question}
        ],
        temperature=0.0
    ).choices[0].message.content
    try:
        req = json.loads(parsed)
    except:
        req = {}
    if req.get("list_metrics"):  return list_metrics(hist)
    if req.get("count_classes"): return count_classes(file,hist)
    if req.get("metadata"):      return answer_metadata(file,hist)
    if not file:
        return hist + [{"role":"assistant","content":"Upload a GeoTIFF first."}], ""
    metrics = req.get("metrics", [])
    if not metrics:
        for code,(f,_d) in metric_definitions.items():
            for syn in synonyms.get(code,[code]):
                if re.search(rf"\b{re.escape(syn)}\b", lower):
                    metrics.append(code)
    if req.get("all_metrics"):
        if req.get("level") == "landscape":
            metrics = [m for m in metric_definitions if m not in class_only]
        elif req.get("level") == "class":
            metrics = list(class_only)
    level = req.get("level","both")
    if not metrics:
        return llm_fallback(hist)
    if level == "landscape":
        return compute_landscape_only( file, metrics, hist)
    if level == "class":
        return compute_class_only( file, metrics, hist)
    return compute_multiple_metrics(file, metrics, hist)

# --- UI layout & launch ---
initial_history = [{"role":"assistant","content":"👋 Hi! Upload a GeoTIFF to begin."}]
with gr.Blocks(title="Spatchat") as iface:
    file_input = gr.File(label="Upload GeoTIFF", type="filepath")
    raster_out = gr.Plot(label="Raster Preview")
    clear_btn  = gr.Button("Clear Raster")
    chatbot    = gr.Chatbot(value=initial_history, type="messages")
    txt_in     = gr.Textbox(placeholder="e.g. calculate edge density", lines=1)
    clr_chat   = gr.Button("Clear Chat")

    file_input.change(preview_raster,    file_input, raster_out)
    file_input.change(notify_upload,     [file_input,chatbot], chatbot)
    clear_btn.click(clear_raster,       None, [file_input,raster_out])
    txt_in.submit(analyze_raster,       [file_input,txt_in,chatbot], [chatbot,txt_in])
    clr_chat.click(lambda:initial_history, None, chatbot)

iface.launch()
