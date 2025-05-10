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
    # Build initial chat messages
    messages = [
        {"role": "system", "content": (
            "You are Spatchat, a helpful assistant that explains and calculates landscape metrics from raster files.\n"
            "If the user explicitly requests landscape metrics, use pylandstats to calculate them.\n"
            "Otherwise, simply acknowledge the upload without precomputing metrics.\n"
            "If the question is too vague, ask the user to clarify."
        )},
        *history,
        {"role": "user", "content": question}
    ]

    # Acknowledge uploaded raster
    if file is not None:
        messages.insert(1, {
            "role": "system",
            "content": "A raster file has been uploaded and is available for analysis."
        })

    # Lowercase question for pattern matching
    lower_q = question.lower()

    # If the user asks for the list of available metrics, return a categorized catalog
    if re.search(r"\b(list|available|which).*metrics\b", lower_q):
        # Define categories and their metric keys
        categories = {
            "Landscape metrics": ["contag", "shdi", "shei", "mesh", "lsi", "tca"],
            "Class metrics": ["pland", "np", "pd", "lpi", "total_edge", "edge_density"],
            "Patch metrics": ["area", "perim", "para", "shape", "frac", "enn", "core", "nca", "cai"]
        }
        # Build catalog text
        catalog_sections = []
        for cat, keys in categories.items():
            lines = [f"- **{metric_definitions[k][0]}** (`{k}`): {metric_definitions[k][1]}" for k in keys]
            catalog_sections.append(f"{cat}:\n" + "\n".join(lines))
        catalog_text = "\n\n".join(catalog_sections)
        response_text = "Here are the available metrics by category:\n\n" + catalog_text
        return history + [{"role": "assistant", "content": response_text}]

    # Detect if the user is asking for metrics calculations
    metric_keys = list(metric_definitions.keys())
    calc_terms = ['calculate', 'metrics', 'what is', 'show', 'compute', 'density', 'number']
    pattern = re.compile(r"\b(" + "|".join(metric_keys + calc_terms) + r")\b", re.IGNORECASE)
    want_metrics = bool(pattern.search(lower_q))

    # Determine specification of class or landscape level
    want_class = bool(re.search(r"\bclass\b", lower_q))
    want_land = bool(re.search(r"\blandscape\b", lower_q))
    if want_metrics and not (want_class or want_land):
        want_class = want_land = True

    response_text = None
    try:
        if file is not None and want_metrics:
            # Read CRS & native resolution
            with rasterio.open(file.name) as src:
                crs = src.crs
                x_res, y_res = src.res

            # Compute metrics with true cell-size
            landscape = Landscape(
                file.name,
                nodata=0,
                resolution=x_res
            )
            df_patch = landscape.compute_patch_metrics_df()
            df_class = landscape.compute_class_metrics_df()
            df_land  = landscape.compute_landscape_metrics_df()

            # Landscape-level string
            land_metrics = (
                f"Patch Density: {df_land.get('patch_density',[float('nan')])[0]:.4f}\n"
                f"Edge Density: {df_land.get('edge_density',[float('nan')])[0]:.4f}\n"
                f"Total Core Area: {df_land.get('total_core_area',[float('nan')])[0]:.4f}\n"
            )
            # Class-level table
            class_metrics = df_class.to_string(index=False)

            # Combine per user spec
            report_parts = []
            if want_land:
                report_parts.append("**Landscape-level metrics:**\n" + land_metrics)
            if want_class:
                report_parts.append("**Class-level metrics:**\n" + class_metrics)

            real_metrics = (
                f"CRS: {crs}\n"
                f"Native cell-size: {x_res:.2f} × {y_res:.2f} (CRS units)\n\n"
                + "\n\n".join(report_parts)
            )

            messages[-1]['content'] += f"\n\nHere are real metrics from the uploaded raster:\n{real_metrics}"

        # Call the LLM
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=messages,
            temperature=0.4
        )
        response_text = response.choices[0].message.content

    except Exception as e:
        return history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"⚠️ LLM error: {e}"}
        ]

    # Append to chat history
    return history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": response_text}
    ]



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
            question_input = gr.Textbox(label="Ask Spatchat", placeholder="e.g., Calculate edge density?", lines=1)
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