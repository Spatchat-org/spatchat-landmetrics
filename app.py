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
        # map strings to integer codes if needed
        if raw.dtype.kind in ('U','S','O'):
            raw_str = raw.astype(str)
            uniq = np.unique(raw_str[raw_str != ''])
            name2code = {n: i+1 for i, n in enumerate(uniq)}
            code2name = {c: n for n, c in name2code.items()}
            data = np.zeros_like(raw, dtype=int)
            for n, c in name2code.items():
                data[raw_str == n] = c
            labels = [code2name[c] for c in sorted(code2name)]
            unique_vals = sorted(code2name)
        else:
            data = raw
            unique_vals = np.unique(data[data != 0]).tolist()
            labels = [f"Class {int(v)}" for v in unique_vals]

        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_vals)))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(data, cmap='tab10', interpolation='nearest')
        ax.set_title("Uploaded Raster")
        ax.axis('off')
        handles = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(unique_vals))]
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

# --- Post-upload prompt ---
def on_upload(file, history):
    if file is not None:
        return history + [
            {"role": "user", "content": "<file uploaded>"},
            {"role": "assistant", "content": (
                "Awesome! I can see your raster now. "
                "You can ask me to calculate any landscape metrics, e.g., 'Calculate Edge Density'."
            )}
        ]
    return history

# --- Unified LLM-powered analyzer with string handling ---
def analyze_raster(file, question, history):
    import re
    # append user message
    history = history + [{"role": "user", "content": question}]
    # system prompt
    messages = [
        {"role": "system", "content": (
            "You are Spatchat, a helpful assistant that explains landscape metrics and describes raster properties.\n"
            "Use rasterio for metadata (CRS, resolution, extent, bands, nodata) and pylandstats for metrics.\n"
            "If the user explicitly requests metrics, calculate landscape- and class-level metrics.\n"
            "Otherwise answer questions (e.g., resolution, extent) directly or list metrics on request."
        )},
        *history
    ]
    if file is not None:
        messages.insert(1, {"role": "system", "content": "A raster file has been uploaded and is available for analysis."})

    lower_q = question.lower()

    # fast-path: resolution
    if file is not None and re.search(r"\bresolution\b", lower_q):
        with rasterio.open(file.name) as src:
            x_res, y_res = src.res
            # use linear_units instead of axis_info
            unit = getattr(src.crs, 'linear_units', 'unit')
        return history + [{"role": "assistant", "content": f"The native resolution is {x_res:.2f} × {y_res:.2f} {unit} per pixel."}]

    # list metrics
    if re.search(r"\b(list|available|which).*metrics\b", lower_q):
        cats = {
            "Landscape": ["contag","shdi","shei","mesh","lsi","tca"],
            "Class": ["pland","np","pd","lpi","total_edge","edge_density"],
            "Patch": ["area","perim","para","shape","frac","enn","core","nca","cai"]
        }
        lines = []
        for cat, keys in cats.items():
            lines.append(f"**{cat}-level metrics:**")
            for k in keys:
                lines.append(f"- {metric_definitions[k][0]} (`{k}`): {metric_definitions[k][1]}")
        return history + [{"role": "assistant", "content": "Here are available metrics:\n" + "\n".join(lines)}]

    # detect compute
    if re.search(r"\b(calculate|compute|metrics|density|number|what is|show)\b", lower_q):
        with rasterio.open(file.name) as src:
            raw = src.read(1)
            crs = src.crs
            x_res, y_res = src.res
            extent = src.bounds
            bands = src.count
            nodata = src.nodata
        # map strings
        if raw.dtype.kind in ('U','S','O'):
            raw_str = raw.astype(str)
            uniq = np.unique(raw_str[raw_str!=''])
            name2code = {n:i+1 for i,n in enumerate(uniq)}
            code2name = {c:n for n,c in name2code.items()}
            arr = np.zeros_like(raw, dtype=int)
            for n,c in name2code.items():
                arr[raw_str==n] = c
        else:
            arr = raw
            code2name = {int(v):f'Class {int(v)}' for v in np.unique(arr[arr!=0])}
        # pass res instead of resolution
        landscape = Landscape(arr, res=x_res, nodata=0)
        df_land = landscape.compute_landscape_metrics_df()
        df_class = landscape.compute_class_metrics_df()
        land_txt = f"Patch Density: {df_land['patch_density'][0]:.4f}\nEdge Density: {df_land['edge_density'][0]:.4f}\n"
        df_class.insert(0,'class_name', df_class.index.to_series().map(code2name))
        df_class = df_class.reset_index(drop=True)
        class_txt = df_class.to_string(index=False)
        header = f"CRS: {crs}\nResolution: {x_res:.2f}×{y_res:.2f}\nExtent: {extent}\nBands: {bands}\nNoData: {nodata}\n\n"
        return history + [{"role": "assistant", "content": header + f"**Landscape-level metrics:**\n{land_txt}\n**Class-level metrics:**\n{class_txt}"}]

    # fallback to LLM
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages,
        temperature=0.4
    )
    return history + [{"role": "assistant", "content": resp.choices[0].message.content}]


# --- UI layout ---
with gr.Blocks(title="Spatchat") as iface:
    # --- Logo and Header ---
    gr.HTML("""
        <head>
            <link rel="icon" type="image/png" href="file=logo1.png">
        </head>
    """ )
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
    """ )
    gr.Markdown("## 🌲 Spatchat: Landscape Metrics Assistant")
    gr.HTML('''
    <div style="margin-top: -10px; margin-bottom: 15px;">
      <input type="text" value="https://spatchat.org/browse/?room=landmetrics" readonly style="width: 50%; padding: 5px; background-color: #f8f8f8; color: #222; font-weight: 500; border: 1px solid #ccc; border-radius: 4px;">
      <button onclick="navigator.clipboard.writeText(this.previousElementSibling.value)" style="padding: 5px 10px; background-color: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer;">
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
    """ )

    # Layout: two columns
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload GeoTIFF", type="filepath")
            raster_output = gr.Plot(label="Raster Preview", visible=True)
            clear_raster_button = gr.Button("Clear Raster")
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(
                label="Spatchat Dialog", 
                type="messages", 
                value=[
                    {"role": "assistant", "content": "Hi, I am Spatchat. I can help you explore your raster—ask about CRS, resolution, extent, or calculate metrics. Please upload a raster to begin."}
                ]
            )
            question_input = gr.Textbox(label="Ask Spatchat", placeholder="e.g., Calculate edge density?", lines=1)
            ask_button = gr.Button("Ask")
            clear_button = gr.Button("Clear Chat")

    # Event bindings
    file_input.change(fn=preview_raster, inputs=file_input, outputs=raster_output)
    file_input.change(fn=on_upload, inputs=[file_input, chatbot], outputs=chatbot)
    clear_raster_button.click(fn=clear_raster, inputs=None, outputs=[file_input, raster_output])
    question_input.submit(fn=analyze_raster, inputs=[file_input, question_input, chatbot], outputs=chatbot)
    ask_button.click(fn=analyze_raster, inputs=[file_input, question_input, chatbot], outputs=chatbot)
    question_input.submit(fn=lambda: "", inputs=None, outputs=question_input)
    ask_button.click(fn=lambda: "", inputs=None, outputs=question_input)
    clear_button.click(fn=lambda: [], inputs=None, outputs=chatbot)

iface.launch()
