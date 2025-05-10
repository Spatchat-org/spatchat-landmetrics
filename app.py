import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
import numpy as np
from pylandstats import Landscape
import re
import pandas as pd
from dotenv import load_dotenv
import os
from together import Together

# --- LLM setup ---
load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# --- Metric glossary ---
metric_definitions = {
    "pland": ("Proportion of Landscape (PLAND)", "Percentage of landscape comprised by the class."),
    "np": ("Number of Patches (NP)", "Total number of patches for the class or landscape."),
    "pd": ("Patch Density (PD)", "Number of patches per 100 hectares."),
    "lpi": ("Largest Patch Index (LPI)", "Percentage of total landscape made up by the largest patch."),
    "te": ("Total Edge (TE)", "Total length of all patch edges."),
    "ed": ("Edge Density (ED)", "Edge length per hectare."),
    "lsi": ("Landscape Shape Index (LSI)", "Overall shape complexity of the landscape."),
    "tca": ("Total Core Area (TCA)", "Sum of all core areas in the landscape."),
    "mesh": ("Effective Mesh Size (MESH)", "Average size of patches after accounting for edge and fragmentation."),
    "contag": ("Contagion Index (CONTAG)", "Clumpiness of patches — higher means more aggregated."),
    "shdi": ("Shannon Diversity Index (SHDI)", "Diversity of patch types."),
    "shei": ("Shannon Evenness Index (SHEI)", "Evenness of patch distribution."),
    "area": ("Patch Area (AREA)", "Area of each individual patch."),
    "perim": ("Patch Perimeter (PERIM)", "Perimeter of each patch."),
    "para": ("Perimeter-Area Ratio (PARA)", "Ratio of perimeter to area for each patch."),
    "shape": ("Shape Index (SHAPE)", "Shape complexity of each patch."),
    "frac": ("Fractal Dimension (FRAC)", "Fractal complexity of each patch shape."),
    "enn": ("Euclidean Nearest Neighbor Distance (ENN)", "Distance to nearest patch of same class."),
    "core": ("Core Area (CORE)", "Interior area of a patch excluding edges."),
    "nca": ("Number of Core Areas (NCA)", "Total core regions in the landscape."),
    "cai": ("Core Area Index (CAI)", "Proportion of core area to total patch area."),
    "edge_density": ("Edge Density (ED)", "Edge length per hectare."),
    "total_edge": ("Total Edge (TE)", "Total length of all patch edges.")
}

# --- Raster preview with class-name mapping ---
def preview_raster(file):
    try:
        with rasterio.open(file.name) as src:
            raw = src.read(1)
        # map string values to integer codes
        if raw.dtype.kind in ('U','S','O'):
            raw_str = raw.astype(str)
            uniq = np.unique(raw_str[raw_str!=''])
            name2code = {n:i+1 for i,n in enumerate(uniq)}
            code2name = {c:n for n,c in name2code.items()}
            data = np.zeros_like(raw, dtype=int)
            for n,c in name2code.items():
                data[raw_str==n] = c
            labels = [code2name[c] for c in sorted(code2name)]
            vals = sorted(code2name)
        else:
            data = raw
            vals = np.unique(data[data!=0]).tolist()
            labels = [f"Class {int(v)}" for v in vals]
        colors = plt.cm.tab10(np.linspace(0,1,len(vals)))
        fig,ax = plt.subplots(figsize=(5,5))
        ax.imshow(data, cmap='tab10', interpolation='nearest')
        ax.axis('off')
        handles = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(vals))]
        ax.legend(handles=handles, loc='lower left', fontsize='small', frameon=True)
        return fig
    except Exception:
        fig,ax = plt.subplots(figsize=(5,5))
        ax.text(0.5,0.5,'🗂️ No raster loaded.', ha='center', va='center', color='gray')
        ax.axis('off')
        return fig

# --- Clear raster ---
def clear_raster():
    fig,ax = plt.subplots(figsize=(5,5))
    ax.text(0.5,0.5,'🗂️ No raster loaded.', ha='center', va='center', color='gray')
    ax.axis('off')
    return None, gr.update(value=fig, visible=True)

# --- Post-upload prompt ---
def on_upload(file, history):
    if file is not None:
        return history + [
            {"role":"user","content":"<file uploaded>"},
            {"role":"assistant","content":"Awesome! I can see your raster now. You can ask me to calculate any landscape metrics, e.g., 'Calculate Edge Density'."}
        ]
    return history

# --- Analyzer ---
def analyze_raster(file, question, history):
    history = history + [{"role":"user","content":question}]
    messages = [
        {"role":"system","content":(
            "You are Spatchat, a helpful assistant that explains landscape metrics and describes raster properties."
            " Use rasterio for metadata and pylandstats for metrics."
        )},
        *history
    ]
    if file is not None:
        messages.insert(1,{"role":"system","content":"A raster file has been uploaded and is available for analysis."})
    lower_q = question.lower()
    # metadata path
    if file and re.search(r"\b(resolution|crs|extent|bands|nodata)\b", lower_q):
        with rasterio.open(file.name) as src:
            x_res,y_res = src.res
            crs = src.crs
            ext = src.bounds
            bcount = src.count
            nod = src.nodata
            unit = getattr(src.crs,'linear_units','unit')
        resp = []
        if 'resolution' in lower_q:
            resp.append(f"Resolution: {x_res:.2f}×{y_res:.2f} {unit}")
        if 'crs' in lower_q:
            resp.append(f"CRS: {crs}")
        if 'extent' in lower_q:
            resp.append(f"Extent: {ext}")
        if 'band' in lower_q:
            resp.append(f"Bands: {bcount}")
        if 'nodata' in lower_q:
            resp.append(f"NoData: {nod}")
        return history + [{"role":"assistant","content":"; ".join(resp)}]
    # metrics catalog
    if re.search(r"\b(list|available|which).*metrics\b", lower_q):
        cats={"Landscape":["contag","shdi","shei","mesh","lsi","tca"],"Class":["pland","np","pd","lpi","total_edge","edge_density"],"Patch":["area","perim","para","shape","frac","enn","core","nca","cai"]}
        txt=[]
        for lvl,keys in cats.items():
            txt.append(f"{lvl}-level metrics:")
            for k in keys:
                txt.append(f"- {metric_definitions[k][0]} (`{k}`)")
        return history + [{"role":"assistant","content":"\n".join(txt)}]
    # compute metrics
    if re.search(r"\b(calculate|compute|metrics|density|number|show)\b", lower_q):
        with rasterio.open(file.name) as src:
            raw=src.read(1); x_res,y_res=src.res; crs=src.crs; ext=src.bounds; bcount=src.count; nod=src.nodata
        if raw.dtype.kind in ('U','S','O'):
            rs=raw.astype(str); uniq=np.unique(rs[rs!='']); n2c={n:i+1 for i,n in enumerate(uniq)}; c2n={c:n for n,c in n2c.items()}; arr=np.zeros_like(raw,dtype=int)
            for n,c in n2c.items(): arr[rs==n]=c
        else:
            arr=raw; c2n={int(v):f"Class {int(v)}" for v in np.unique(arr[arr!=0])}
        ls=Landscape(arr,res=(x_res,y_res),nodata=0)
        dfL,dfC=ls.compute_landscape_metrics_df(),ls.compute_class_metrics_df()
        land_txt=f"Patch Density: {dfL['patch_density'][0]:.4f}\nEdge Density: {dfL['edge_density'][0]:.4f}" 
        dfC.insert(0,'class_name',dfC.index.to_series().map(c2n)); cls_txt=dfC.reset_index(drop=True).to_string(index=False)
        hdr=f"CRS: {crs}\nRes: {x_res:.2f}×{y_res:.2f}\nExtent: {ext}\nBands: {bcount}\nNoData: {nod}\n\n"
        body=f"Landscape metrics:\n{land_txt}\n\nClass metrics:\n{cls_txt}"
        return history+[{"role":"assistant","content":hdr+body}]
    # fallback
    resp=client.chat.completions.create(model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",messages=messages)
    return history+[{"role":"assistant","content":resp.choices[0].message.content}]

# --- UI layout ---
with gr.Blocks(title="Spatchat") as iface:
    gr.HTML("""
    <head><link rel=\"icon\" type=\"image/png\" href=\"file=logo1.png\"></head>
    """)
    gr.Image(value="logo/logo_long1.png",show_label=False,show_download_button=False,show_share_button=False,type="filepath",elem_id="logo-img")
    gr.HTML("""
    <style>#logo-img img{height:90px;margin:10px 50px 10px 10px;border-radius:6px;}</style>
    """)
    gr.Markdown("## 🌲 Spatchat: Landscape Metrics Assistant")
    gr.HTML('''
    <div style="margin:-10px 0 15px;">
      <input value="https://spatchat.org/browse/?room=landmetrics" readonly style="width:50%;padding:5px;background:#f8f8f8;color:#222;border:1px solid #ccc;border-radius:4px;">
      <button onclick="navigator.clipboard.writeText(this.previousElementSibling.value)" style="padding:5px 10px;background:#007BFF;color:#fff;border:none;border-radius:4px;cursor:pointer;">📋Copy</button>
      <span style="margin-left:10px;font-size:14px;">Share: <a href="https://twitter.com/intent/tweet?...">🐦</a> | <a href="https://facebook.com/sharer?...">📘</a></span>
    </div>
    ''')
    gr.Markdown("""
    <div style="font-size:14px;">©2025 Ho Yi Wan & Logan Hysen.<br>If you use Spatchat, please cite: Wan, H.Y. & Hysen, L. (2025).</div>
    """)
    with gr.Row():
        with gr.Column(scale=1):
            file_input=gr.File(label="Upload GeoTIFF",type="filepath")
            raster_output=gr.Plot(label="Raster Preview")
            clear_btn=gr.Button("Clear Raster")
        with gr.Column(scale=1):
            chatbot=gr.Chatbot(value=[{"role":"assistant","content":"Hi, I'm Spatchat. Upload a raster to begin."}],type="messages",label="Spatchat Dialog")
            txt=gr.Textbox(label="Ask Spatchat",placeholder="e.g., Calculate edge density?",lines=1)
            ask=gr.Button("Ask")
            clr=gr.Button("Clear Chat")
    file_input.change(preview_raster,inputs=file_input,outputs=raster_output)
    file_input.change(on_upload,inputs=[file_input,chatbot],outputs=chatbot)
    clear_btn.click(clear_raster,inputs=None,outputs=[file_input,raster_output])
    txt.submit(analyze_raster,inputs=[file_input,txt,chatbot],outputs=chatbot)
    ask.click(analyze_raster,inputs=[file_input,txt,chatbot],outputs=chatbot)
    clr.click(lambda:[],inputs=None,outputs=chatbot)
iface.launch()
