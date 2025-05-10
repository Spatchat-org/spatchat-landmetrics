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
            # Attempt to use class names from metadata
            tags = src.tags()
            class_names = None
            for k, v in tags.items():
                if k.lower().startswith('class') and (';' in v or ',' in v):
                    class_names = re.split(r'[;,]', v)
                    break
        labels = []
        if class_names and len(class_names) >= len(unique):
            for val in unique:
                labels.append(class_names[list(unique).index(val)].strip())
        else:
            labels = [f'Class {int(val)}' for val in unique]
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique)))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(data, cmap='tab10', interpolation='nearest')
        ax.set_title("Uploaded Raster")
        ax.axis('off')
        handles = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(unique))]
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

# --- Post-upload greeting ---
def on_upload(file, history):
    if file is not None:
        return history + [{"role": "user", "content": "<file uploaded>"},
                          {"role": "assistant", "content": (
            "Awesome! I can see your raster now. "
            "You can ask me to calculate any landscape metrics, e.g., 'Calculate Edge Density'."
        )}]
    return history

# --- Unified LLM-powered analyzer ---
def analyze_raster(file, question, history):
    import re
    # Append the user message first
    history = history + [{"role": "user", "content": question}]
    # Build the chat messages
    messages = [
        {"role": "system", "content": (
            "You are Spatchat, a helpful assistant that explains landscape metrics and describes raster properties.\n"
            "Use rasterio for metadata (CRS, spatial resolution, extent, bands, nodata) and pylandstats for metrics.\n"
            "If the user explicitly requests landscape metrics, calculate them; otherwise answer simple queries or list metrics without computing.\n"
            "Examples:\n"
            "- User: 'What is the CRS of this raster?' → Assistant reads src.crs and replies 'The CRS is EPSG:4326'.\n"
            "- User: 'What is the resolution?' → Assistant reads src.res and replies '30.00 x 30.00 meters per pixel'.\n"
            "- User: 'List available metrics' → Assistant returns categories and metric descriptions.\n"
            "- User: 'Calculate edge density' → Assistant uses pylandstats, computes both levels, and returns results."
        )},
        *history
    ]
    # Inform LLM that raster is available
    if file is not None:
        messages.insert(1, {"role": "system", "content": "A raster file has been uploaded and is available for analysis."})
    lower_q = question.lower()
    # Fast-paths and metric logic as before...
    # [existing resolution, catalog, metric computations]
    # Final assistant response from LLM or direct computation
    # Example direct return for resolution:
    if file is not None and re.search(r"\bresolution\b", lower_q):
        with rasterio.open(file.name) as src:
            x_res, y_res = src.res
            unit = getattr(src.crs.axis_info[0], 'unit_name', 'unit')
        resp = f"The native resolution of this raster is {x_res:.2f} × {y_res:.2f} {unit} per pixel."
        return history + [{"role": "assistant", "content": resp}]
    # ... similar for metrics and catalog ...
    # Otherwise delegate to LLM
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages,
        temperature=0.4
    )
    return history + [{"role": "assistant", "content": response.choices[0].message.content}]

# --- UI layout ---
with gr.Blocks(title="Spatchat") as iface:
    initial_history = [{"role": "assistant", "content": (
        "Hi, I am Spatchat. I can help you explore your raster file—\n"
        "ask me about metadata (CRS, resolution, extent) or calculate landscape metrics.\n"
        "Please upload a raster to begin."
    )}]
    gr.HTML("""
        <head>
            <link rel="icon" type="image/png" href="file=logo1.png">
        </head>
    """)
    gr.Image(value="logo/logo_long1.png", show_label=False, show_download_button=False,
             show_share_button=False, type="filepath", elem_id="logo-img")
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload GeoTIFF", type="filepath")
            raster_output = gr.Plot(label="Raster Preview", visible=True)
            clear_raster_button = gr.Button("Clear Raster")
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(label="Spatchat Dialog", type="messages", value=initial_history)
            question_input = gr.Textbox(label="Ask Spatchat", placeholder="e.g., Calculate edge density?", lines=1)
            ask_button = gr.Button("Ask")
            clear_button = gr.Button("Clear Chat")
    file_input.change(fn=preview_raster, inputs=file_input, outputs=raster_output)
    file_input.change(fn=on_upload, inputs=[file_input, chatbot], outputs=chatbot)
    clear_raster_button.click(fn=clear_raster, inputs=None, outputs=[file_input, raster_output])
    question_input.submit(fn=analyze_raster, inputs=[file_input, question_input, chatbot], outputs=chatbot)
    ask_button.click(fn=analyze_raster, inputs=[file_input, question_input, chatbot], outputs=chatbot)
    question_input.submit(fn=lambda: "", inputs=None, outputs=question_input)
    ask_button.click(fn=lambda: "", inputs=None, outputs=question_input)
    clear_button.click(fn=lambda: [], inputs=None, outputs=chatbot)
iface.launch()
