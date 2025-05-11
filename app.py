import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
from rasterio.plot import show
import numpy as np
from pylandstats import Landscape
import re
import pandas as pd
from itertools import product
from io import StringIO
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
            unique = np.unique(data[data != 0])
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique)))
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(data, cmap='tab10', interpolation='nearest')
            ax.set_title("Uploaded Raster")
            ax.axis('off')
            handles = [mpatches.Patch(color=colors[i], label=f'Class {int(val)}') for i, val in enumerate(unique)]
            ax.legend(handles=handles, loc='lower left', fontsize='small', frameon=True)
            return fig
    except Exception:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, "🗂️ No raster loaded.", fontsize=12, ha='center', va='center', color='gray')
        ax.set_title("Raster Preview", fontsize=14, color='dimgray')
        ax.axis('off')
        return fig

# --- Clear raster ---
def clear_raster():
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.text(0.5, 0.5, "🗂️ No raster loaded.", fontsize=12, ha='center', va='center', color='gray')
    ax.set_title("Raster Preview", fontsize=14, color='dimgray')
    ax.axis('off')
    return None, gr.update(value=fig, visible=True)

# --- Unified LLM-powered analyzer ---
def analyze_raster(file, question, history):
    import re

    # 1. Append the user’s question
    history = history + [{"role": "user", "content": question}]
    lower_q = question.lower()

    # 2. If no raster is loaded, prompt to upload
    if file is None:
        return history + [{
            "role": "assistant",
            "content": "Please upload a GeoTIFF before asking for metadata or metrics."
        }]

    # 3. Read raster metadata once
    with rasterio.open(file.name) as src:
        crs     = src.crs
        x_res,y_res = src.res
        extent  = src.bounds
        bands   = src.count
        nodata  = src.nodata
        unit    = getattr(src.crs, "linear_units", "unit")

    # 4. Metadata queries
    if re.search(r"\b(resolution|crs|extent|bands|nodata)\b", lower_q):
        parts = []
        if "resolution" in lower_q:
            parts.append(f"Resolution: {x_res:.2f} × {y_res:.2f} {unit}")
        if "crs" in lower_q:
            parts.append(f"CRS: {crs}")
        if "extent" in lower_q:
            parts.append(f"Extent: {extent}")
        if "bands" in lower_q:
            parts.append(f"Bands: {bands}")
        if "nodata" in lower_q:
            parts.append(f"NoData: {nodata}")
        return history + [{
            "role": "assistant",
            "content": " ".join(parts)
        }]

    # 5. List available metrics
    if re.search(r"\b(what|which|list|available).*metrics\b", lower_q):
        cats = {
            "Landscape": ["contag","shdi","shei","mesh","lsi","tca"],
            "Class":     ["pland","np","pd","lpi","total_edge","edge_density"],
            "Patch":     ["area","perim","para","shape","frac","enn","core","nca","cai"]
        }
        lines = []
        for lvl, keys in cats.items():
            lines.append(f"**{lvl}-level metrics:**")
            for k in keys:
                name,_ = metric_definitions[k]
                lines.append(f"- {name} (`{k}`)")
        return history + [{
            "role": "assistant",
            "content": "\n".join(lines)
        }]

    # 6. Compute a specific metric if named
    requested = [k for k in metric_definitions if re.search(rf"\b{k}\b", lower_q)]
    if requested:
        key = requested[0]
        landscape = Landscape(file.name, nodata=0)
        df_land   = landscape.compute_landscape_metrics_df()
        df_class  = landscape.compute_class_metrics_df()

        # Landscape-level
        name,_ = metric_definitions[key]
        land_part = ""
        if key in df_land.columns:
            val = df_land[key].iloc[0]
            land_part = f"**Landscape-level {name}:** {val:.4f}\n\n"

        # Class-level
        df_class = df_class.rename_axis("code").reset_index()
        df_class["class_name"] = df_class["code"].map(lambda c: f"Class {int(c)}")
        class_tbl = df_class[["class_name", key]].to_markdown(index=False)

        # Metadata header
        meta = (
            f"CRS: {crs}\n"
            f"Resolution: {x_res:.2f}×{y_res:.2f}\n"
            f"Extent: {extent}\n"
            f"Bands: {bands}\n"
            f"NoData: {nodata}\n\n"
        )

        return history + [{
            "role": "assistant",
            "content": meta + land_part + f"**Class-level {name}:**\n{class_tbl}"
        }]

    # 7. LLM fallback for everything else
    messages = [
        {"role": "system", "content": (
            "You are Spatchat, a helpful assistant that explains landscape metrics and describes raster properties. "
            "Use rasterio and pylandstats when asked to calculate metrics; otherwise reply conversationally."
            "If a raster is uploaded and the user asks for a metric, use pylandstats to calculate it.\n"
            "You can calculate patch metrics, class metrics, and landscape metrics using pylandstats.\n"
            "If the user asks you to calculate a metric or multiple metrics, display the results clearly. You do not need to provide detailed explanations.\n"
            "Be conversational and helpful. If the question is vague, ask the user to clarify."
        )},
        *history
    ]
    # ground in uploaded raster
    messages.insert(1, {"role":"system","content":"A raster file has been uploaded and is available."})

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages,
        temperature=0.4
    ).choices[0].message.content

    return history + [{"role":"assistant","content": response}]

# --- UI layout ---
with gr.Blocks(title="Spatchat") as iface:
    gr.HTML("""
        <head>
            <link rel="icon" type="image/png" href="file=logo1.png">
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
        margin: 10px 50px 10px 10px;  /* top, right, bottom, left */
        border-radius: 6px;
    }
    </style>
    """)
    gr.Markdown("## 🌲 Spatchat: Landscape Metrics Assistant {landmetrics}")
    gr.HTML('''
    <div style="margin-top: -10px; margin-bottom: 15px;">
      <input type="text" value="https://spatchat.org/browse/?room=landmetrics" id="shareLink" readonly style="width: 50%; padding: 5px; background-color: #f8f8f8; color: #222; font-weight: 500; border: 1px solid #ccc; border-radius: 4px;">
      <button onclick="navigator.clipboard.writeText(document.getElementById('shareLink').value)" style="padding: 5px 10px; background-color: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer;">
        📋 Copy Share Link
      </button>
      <div style="margin-top: 10px; font-size: 14px;">
        <b>Share:</b>
        <a href="https://twitter.com/intent/tweet?text=Checkout+Spatchat!&url=https://spatchat.org/browse/?room=landmetrics" target="_blank">🐦 Twitter</a> |
        <a href="https://www.facebook.com/sharer/sharer.php?u=https://spatchat.org/browse/?room=landmetrics" target="_blank">📘 Facebook</a>
      </div>
    </div>
    ''')
    gr.Markdown("""
                <div style="font-size: 14px;">
                © 2025 Ho Yi Wan & Logan Hysen. All rights reserved.<br>
                If you use Spatchat in research, please cite:<br>
                <b>Wan, H.Y.</b> & <b>Hysen, L.</b> (2025). <i>Spatchat: Landscape Metrics Assistant.</i>
                </div>
                """)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload GeoTIFF", type="filepath")
            raster_output = gr.Plot(label="Raster Preview", visible=True)
            clear_raster_button = gr.Button("Clear Raster")
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(label="Spatchat Dialog", type="messages")
            question_input = gr.Textbox(label="Ask Spatchat", placeholder="e.g., What is edge density?", lines=1)
            ask_button = gr.Button("Ask")
            clear_button = gr.Button("Clear Chat")

    file_input.change(fn=preview_raster, inputs=file_input, outputs=raster_output)
    clear_raster_button.click(fn=clear_raster, inputs=None, outputs=[file_input, raster_output])
    question_input.submit(fn=analyze_raster, inputs=[file_input, question_input, chatbot], outputs=chatbot)
    ask_button.click(fn=analyze_raster, inputs=[file_input, question_input, chatbot], outputs=chatbot)
    question_input.submit(fn=lambda: "", inputs=None, outputs=question_input)
    ask_button.click(fn=lambda: "", inputs=None, outputs=question_input)
    clear_button.click(fn=lambda: [], inputs=None, outputs=chatbot)

iface.launch()