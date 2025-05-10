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
            "If the user explicitly requests metrics, calculate both landscape- and class-level.\n"
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
            unit = getattr(src.crs.axis_info[0], 'unit_name', 'unit')
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
    metric_keys = list(metric_definitions.keys())
    if re.search(r"\b(calculate|compute|metrics|density|number|what is|show)\b", lower_q):
        # read raw data and metadata
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
        # compute metrics
        landscape = Landscape(arr, nodata=0, resolution=x_res)
        df_land = landscape.compute_landscape_metrics_df()
        df_class = landscape.compute_class_metrics_df()
        # format
        land_txt = f"Patch Density: {df_land['patch_density'][0]:.4f}\nEdge Density: {df_land['edge_density'][0]:.4f}\n"
        df_class.insert(0, 'class_name', df_class.index.to_series().map(code2name))
        df_class = df_class.reset_index(drop=True)
        class_txt = df_class.to_string(index=False)
        header = f"CRS: {crs}\nResolution: {x_res:.2f}×{y_res:.2f}\nExtent: {extent}\nBands: {bands}\nNoData: {nodata}\n\n"
        body = f"**Landscape-level metrics:**\n{land_txt}\n**Class-level metrics:**\n{class_txt}"
        return history + [{"role": "assistant", "content": header + body}]

    # fallback to LLM
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages,
        temperature=0.4
    )
    return history + [{"role": "assistant", "content": resp.choices[0].message.content}]

# --- UI layout ---
with gr.Blocks(title="Spatchat") as iface:
    initial_history = [{"role": "assistant", "content": (
        "Hi, I am Spatchat. I can help you explore your raster—ask about CRS, resolution, extent or calculate landscape metrics.\n"
        "Please upload a raster to begin."
    )}]

    file_input = gr.File(label="Upload GeoTIFF", type="filepath")
    raster_output = gr.Plot(label="Raster Preview")
    chatbot = gr.Chatbot(value=initial_history, label="Spatchat Dialog")
    question_input = gr.Textbox(label="Ask Spatchat", placeholder="e.g., Calculate edge density?", lines=1)
    ask_button = gr.Button("Ask")
    clear_button = gr.Button("Clear Chat")

    # event bindings
    file_input.change(preview_raster, inputs=file_input, outputs=raster_output)
    file_input.change(on_upload, inputs=[file_input, chatbot], outputs=chatbot)
    clear_button.click(lambda: initial_history, inputs=None, outputs=chatbot)
    ask_button.click(analyze_raster, inputs=[file_input, question_input, chatbot], outputs=chatbot)
    question_input.submit(analyze_raster, inputs=[file_input, question_input, chatbot], outputs=chatbot)
    question_input.submit(lambda: "", None, question_input)

iface.launch()
