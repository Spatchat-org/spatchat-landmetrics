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
    messages = [
        {"role": "system", "content": (
            "You are Spatchat, a helpful assistant that explains and calculates landscape metrics from raster files.\n"
            "If no raster is uploaded, do not attempt to calculate anything.\n"
            "If a raster is uploaded and the user asks for a metric, use pylandstats to calculate it.\n"
            "You can calculate patch metrics, class metrics, and landscape metrics using pylandstats.\n"
            "If the user asks you to calculate a metric or multiple metrics, display the results clearly. You do not need to provide detailed explanations.\n"
            "Be conversational and helpful. If the question is vague, ask the user to clarify."
        )},
        *history,
        {"role": "user", "content": question}
    ]

    # Add grounding system message if raster is uploaded
    if file is not None:
        messages.insert(1, {
            "role": "system",
            "content": "A raster file has been uploaded. You may reference real landscape metrics that were computed using pylandstats."
        })

    try:
        # Inject real metrics into prompt if available
        if file is not None:
            landscape = Landscape(file.name, nodata=0)
            df_patch = landscape.compute_patch_metrics_df()
            df_class = landscape.compute_class_metrics_df()
            df_land = landscape.compute_landscape_metrics_df()

            real_metrics = (
                f"Patch Density: {df_land.get('patch_density', [None])[0]:.4f}\n"
                f"Edge Density: {df_land.get('edge_density', [None])[0]:.4f}\n"
                f"Number of Patches: {df_class['number_of_patches'].sum()}\n"
            )
            messages[-1]['content'] += f"\n\nHere are real metrics from the uploaded raster:\n{real_metrics}"

        # Get LLM response
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=messages,
            temperature=0.4
        ).choices[0].message.content

    except Exception as e:
        return history + [{"role": "user", "content": question}, {"role": "assistant", "content": f"⚠️ LLM error: {e}"}]

    return history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response}
    ]


# --- UI layout ---
with gr.Blocks(title="Spatchat") as iface:
    gr.HTML("""
        <head><link rel="icon" type="image/x-icon" href="favicon.ico"></head>
    """)
    gr.Image(value="logo/logo_long1.png", width=20, show_label=False, show_download_button=False)
    gr.Markdown("## 🌲 Spatchat: Landscape Metric Assistant")

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