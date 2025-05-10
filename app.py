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

# --- Raster preview ---
def preview_raster(file):
    try:
        with rasterio.open(file.name) as src:
            data = src.read(1)
        # Map string categories if present
        if data.dtype.kind in ('U', 'S', 'O'):
            strs = data.astype(str)
            cats = np.unique(strs[strs != ''])
            name2code = {n: i+1 for i, n in enumerate(cats)}
            code2name = {v: k for k, v in name2code.items()}
            arr = np.zeros_like(data, dtype=int)
            for n, c in name2code.items():
                arr[strs == n] = c
            labels = [code2name[i] for i in sorted(code2name)]
        else:
            arr = data
            vals = np.unique(arr[arr != 0])
            labels = [f"Class {int(v)}" for v in vals]
        colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(arr, cmap='tab10', interpolation='nearest')
        ax.axis('off')
        handles = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
        ax.legend(handles=handles, loc='lower left', fontsize='small')
        return fig
    except:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, '🗂️ No raster loaded.', ha='center', va='center', color='gray')
        ax.axis('off')
        return fig

# --- Clear raster ---
def clear_raster():
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.text(0.5, 0.5, '🗂️ No raster loaded.', ha='center', va='center', color='gray')
    ax.axis('off')
    return None, gr.update(value=fig)

# --- Post-upload prompt ---
def on_upload(file, history):
    if file:
        return history + [{"role": "assistant", "content": (
            "Awesome! Raster uploaded. Ask me to calculate any metric, e.g. 'calculate np' or 'what is contag'"
        )}]
    return history

# --- Analyzer ---
def analyze_raster(file, question, history):
    history = history + [{"role": "user", "content": question}]
    lower_q = question.lower()

    # Metadata queries
    if file and re.search(r"\b(resolution|crs|extent|bands|nodata)\b", lower_q):
        with rasterio.open(file.name) as src:
            x_res, y_res = src.res
            crs = src.crs
            ext = src.bounds
            bands = src.count
            nodata = src.nodata
            unit = getattr(src.crs, 'linear_units', 'unit')
        replies = []
        if 'resolution' in lower_q:
            replies.append(f"Resolution: {x_res:.2f} × {y_res:.2f} {unit}")
        if 'crs' in lower_q:
            replies.append(f"CRS: {crs}")
        if 'extent' in lower_q:
            replies.append(f"Extent: {ext}")
        if 'bands' in lower_q:
            replies.append(f"Bands: {bands}")
        if 'nodata' in lower_q:
            replies.append(f"NoData: {nodata}")
        return history + [{"role": "assistant", "content": "; ".join(replies)}], ""

    # List available metrics
    if re.search(r"\b(what|which|list|available).*metrics\b", lower_q):
        lines = []
        for lvl, keys in {"Landscape": ["contag","shdi","shei","mesh","lsi","tca"],
                          "Class": ["pland","np","pd","lpi","total_edge","edge_density"],
                          "Patch": ["area","perim","para","shape","frac","enn","core","nca","cai"]}.items():
            lines.append(f"**{lvl}-level metrics:**")
            for k in keys:
                lines.append(f"- {metric_definitions[k][0]} (`{k}`)")
        return history + [{"role": "assistant", "content": "\n".join(lines)}], ""

    # Compute any requested metric
    if file:
        # detect metrics by code or name
        requested = []
        for key in metric_definitions:
            phrase = key.replace('_', ' ')
            if re.search(rf"\b{key}\b|\b{phrase}\b", lower_q):
                requested.append(key)
        if requested:
            key = requested[0]
            # read and map
            with rasterio.open(file.name) as src:
                raw = src.read(1)
                x_res, y_res = src.res
            if raw.dtype.kind in ('U', 'S', 'O'):
                rs = raw.astype(str)
                cats = np.unique(rs[rs!=''])
                name2code = {n:i+1 for i,n in enumerate(cats)}
                code2name = {v:k for k,v in name2code.items()}
                arr = np.zeros_like(raw, dtype=int)
                for n,c in name2code.items(): arr[rs==n]=c
            else:
                arr = raw
                code2name = {int(v):f"Class {int(v)}" for v in np.unique(arr[arr!=0])}
            ls = Landscape(arr, res=(x_res, y_res), nodata=0)
            df_land = ls.compute_landscape_metrics_df()
            df_class = ls.compute_class_metrics_df()
            val = df_land[key if key!='ed' else 'edge_density'][0]
            df_class.insert(0, 'class_name', df_class.index.to_series().map(code2name))
            tab = df_class[['class_name', key if key!='ed' else 'edge_density']].to_markdown(index=False)
            meta = f"CRS: {crs}\nResolution: {x_res:.2f}×{y_res:.2f}\nExtent: {ext}\nBands: {bands}\nNoData: {nodata}\n\n"
            name = metric_definitions[key][0]
            out = (f"**Landscape-level {name}:** {val:.4f}\n\n"
                   f"**Class-level {name}:**\n{tab}")
            return history + [{"role": "assistant", "content": meta + out}], ""

    # Fallback conversational
    messages = [
        {"role": "system", "content": (
            "You are Spatchat, a helpful assistant explaining landscape metrics and raster properties."
            " Use computational logic when users ask to calculate specific metrics, otherwise reply conversationally without code."
        )},
        *history
    ]
    if file:
        messages.insert(1, {"role": "system", "content": "Raster file is available for analysis."})
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages,
        temperature=0.4
    )
    return history + [{"role": "assistant", "content": resp.choices[0].message.content}], ""

# --- UI layout ---
with gr.Blocks(title="Spatchat") as iface:
    gr.HTML("""
    <head>
        <link rel=\"icon\" type=\"image/png\" href=\"file=logo1.png\">
    </head>
    """)
    gr.Image(
        value="logo/logo_long1.png",
        show_label=False,
        show_download_button=False,
        show_share_button=False,
        type="filepath",
        elem_id="logo-img"
    )
    gr.HTML("""
    <style>
    #logo-img img {
        height: 90px;
        margin: 10px 50px 10px 10px;
        border-radius: 6px;
    }
    </style>
    """)
    gr.Markdown("## 🌲 Spatchat: Landscape Metrics Assistant")
    gr.HTML('''
    <div style="margin-top:-10px;margin-bottom:15px;">
      <input id="shareLink" value="https://spatchat.org/browse/?room=landmetrics" readonly style="width:50%;padding:5px;background:#f8f8f8;border:1px solid #ccc;border-radius:4px;">
      <button onclick="navigator.clipboard.writeText(document.getElementById('shareLink').value)" style="padding:5px 10px;background:#007BFF;color:#fff;border:none;border-radius:4px;cursor:pointer;">📋 Copy Link</button>
      <span style="margin-left:10px;font-size:14px;">Share: <a href="https://twitter.com/intent/tweet?url=https://spatchat.org/browse/?room=landmetrics">🐦</a> | <a href="https://www.facebook.com/sharer/sharer.php?u=https://spatchat.org/browse/?room=landmetrics">📘</a></span>
    </div>
    ''')
    gr.Markdown("""
    <div style="font-size:14px;">© 2025 Ho Yi Wan & Logan Hysen. If used in research, please cite: Wan, H.Y. & Hysen, L. (2025).</div>
    """
    )
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload GeoTIFF", type="filepath")
            raster_plot = gr.Plot(label="Raster Preview")
            clear_btn = gr.Button("Clear Raster")
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(value=[{"role":"assistant","content":"Hi! Upload a raster to begin."}], type="messages", label="Spatchat Dialog")
            question = gr.Textbox(label="Ask Spatchat", placeholder="e.g., calculate edge density?", lines=1)
            ask_btn = gr.Button("Ask")
            clear_chat = gr.Button("Clear Chat")
    file_input.change(preview_raster, inputs=file_input, outputs=raster_plot)
    file_input.change(on_upload, inputs=[file_input, chatbot], outputs=chatbot)
    clear_btn.click(clear_raster, inputs=None, outputs=[file_input, raster_plot])
    question.submit(analyze_raster, inputs=[file_input, question, chatbot], outputs=[chatbot, question])
    ask_btn.click(analyze_raster, inputs=[file_input, question, chatbot], outputs=[chatbot, question])
    clear_chat.click(lambda: [{"role":"assistant","content":"Hi! Upload a raster to begin."}], inputs=None, outputs=chatbot)
iface.launch()
