import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
import numpy as np
import re
import yaml
from pylandstats import Landscape
from together import Together
import os
from dotenv import load_dotenv

# --- LLM setup ---
load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# --- Load Glossary from YAML (with fallback) ---
try:
    for fname in ("spatchat_glossary.yaml", "spatchat_glossary.yml", "spatchat_glossary.txt"):
        if os.path.exists(fname):
            with open(fname) as f:
                glossary = yaml.safe_load(f)
            break
    if not isinstance(glossary, dict):
        raise ValueError("glossary root is not a mapping")
except Exception as e:
    print(f"⚠️ Warning loading glossary: {e}")
    glossary = {
        "pland": {"name": "Proportion of Landscape (PLAND)",
                  "definition": "Percentage of landscape comprised by the class.",
                  "units": "%",
                  "interpretation": "Higher means more area of that class."},
        "np":   {"name": "Number of Patches (NP)",
                  "definition": "Total number of patches.",
                  "units": "count",
                  "interpretation": "Higher means more fragmentation."},
        # … add other minimal defaults as needed …
    }

# Build a flat metric_definitions mapping code → info dict
metric_definitions = glossary  # each value has keys: name, definition, units, interpretation

# --- Raster preview function ---
def preview_raster(file):
    try:
        with rasterio.open(file.name) as src:
            data = src.read(1)
        unique = np.unique(data[data != src.nodata])
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique)))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(data, cmap='tab10', interpolation='nearest')
        ax.set_title("Uploaded Raster")
        ax.axis('off')
        handles = [
            mpatches.Patch(color=colors[i],
                           label=metric_definitions.get(str(int(val)), {"name": f"Class {int(val)}"})["name"])
            for i, val in enumerate(unique)
        ]
        ax.legend(handles=handles, loc='lower left', fontsize='small', frameon=True)
        return fig
    except Exception:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, "🗂️ No raster loaded.", fontsize=12,
                ha='center', va='center', color='gray')
        ax.set_title("Raster Preview", fontsize=14, color='dimgray')
        ax.axis('off')
        return fig

def clear_raster():
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.text(0.5, 0.5, "🗂️ No raster loaded.", fontsize=12,
            ha='center', va='center', color='gray')
    ax.set_title("Raster Preview", fontsize=14, color='dimgray')
    ax.axis('off')
    return None, gr.update(value=fig, visible=True)

# --- Helper functions for intent branches ---
def answer_metadata(file, history):
    with rasterio.open(file.name) as src:
        crs     = src.crs
        x_res,y_res = src.res
        extent  = src.bounds
        bands   = src.count
        nodata  = src.nodata
        unit    = getattr(src.crs, "linear_units", "unit")
    parts = [
        f"CRS: {crs}",
        f"Resolution: {x_res:.2f} × {y_res:.2f} {unit}",
        f"Extent: {extent}",
        f"Bands: {bands}",
        f"NoData value: {nodata}"
    ]
    return history + [{"role":"assistant","content":"\n".join(parts)}]

def list_metrics(history):
    # group by level
    by_level = {"Landscape": [], "Class": [], "Patch": []}
    for code, info in metric_definitions.items():
        lvl = info.get("level", "")
        # allow combined levels like "Class / Landscape"
        for L in by_level:
            if L.lower() in lvl.lower():
                by_level[L].append((code, info["name"]))
    lines = []
    for L, items in by_level.items():
        lines.append(f"**{L}-level metrics:**")
        for code, name in sorted(items):
            lines.append(f"- {name} (`{code}`)")
        lines.append("")  # blank line
    return history + [{"role":"assistant","content":"\n".join(lines).strip()}]

def compute_metric(file, key, history):
    landscape = Landscape(file.name, nodata=0)
    df_land   = landscape.compute_landscape_metrics_df()
    df_class  = landscape.compute_class_metrics_df()
    info      = metric_definitions.get(key, {})
    name      = info.get("name", key)

    land_part = ""
    if key in df_land.columns:
        val = df_land[key].iloc[0]
        land_part = f"**Landscape-level {name}:** {val:.4f}\n\n"

    df_c = df_class.rename_axis("code").reset_index()
    df_c["class_name"] = df_c["code"].map(lambda c: f"Class {int(c)}")
    tbl = df_c[["class_name", key]].to_markdown(index=False)

    return history + [{
        "role":"assistant",
        "content": land_part + f"**Class-level {name}:**\n{tbl}"
    }]

def explain_metric(key, history):
    info = metric_definitions.get(key, {})
    text = (
        f"**{info.get('name','')}**\n\n"
        f"{info.get('definition','')}\n\n"
        f"Units: {info.get('units','')}\n\n"
        f"Interpretation: {info.get('interpretation','')}"
    )
    return history + [{"role":"assistant","content":text}]

# --- Main analyzer with LLM-based intent classification ---
def analyze_raster(file, question, history):
    # 1) append user message
    history = history + [{"role":"user","content":question}]

    # 2) classify intent with a first LLM call
    classification_prompt = [
        {"role":"system","content":(
            "You are Spatchat’s intent classifier. Respond with exactly one label:\n"
            "METADATA – asks about CRS, resolution, extent, bands, or nodata\n"
            "LIST_METRICS – asks to list all available metrics\n"
            "COMPUTE:<metric_code> – asks to calculate a known metric (e.g. COMPUTE:edge_density)\n"
            "EXPLAIN:<metric_code> – asks to explain what a metric means\n"
            "FALLBACK – for anything else"
        )},
        *history
    ]
    intent = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=classification_prompt,
        temperature=0
    ).choices[0].message.content.strip()

    # 3) require raster for non-fallback intents
    if file is None and intent != "FALLBACK":
        return history + [{
            "role":"assistant",
            "content":"Please upload a GeoTIFF before asking metadata or metrics."
        }]

    # 4) dispatch to the right handler
    if intent == "METADATA":
        return answer_metadata(file, history)
    if intent == "LIST_METRICS":
        return list_metrics(history)
    if intent.startswith("COMPUTE:"):
        key = intent.split(":",1)[1]
        return compute_metric(file, key, history)
    if intent.startswith("EXPLAIN:"):
        key = intent.split(":",1)[1]
        return explain_metric(key, history)

    # 5) fallback to full LLM chat
    fallback_prompt = [
        {"role":"system","content":(
            "You are Spatchat, a helpful assistant. Use rasterio to fetch metadata, "
            "use pylandstats to calculate metrics, and refer to the glossary for metric definitions. "
            "Otherwise be conversational and friendly."
        )},
        *history
    ]
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=fallback_prompt,
        temperature=0.4
    ).choices[0].message.content

    return history + [{"role":"assistant","content":response}]

# --- UI layout ---
initial_history = [
    {"role":"assistant","content":
     "👋 Hi! I’m Spatchat. Upload a GeoTIFF to begin—then ask me for CRS, resolution, or any landscape metric."}
]

with gr.Blocks(title="Spatchat") as iface:
    # favicon & logo
    gr.HTML("""<head><link rel="icon" href="file=logo1.png"></head>""")
    gr.Image(value="logo/logo_long1.png", type="filepath", show_label=False,
             show_download_button=False, show_share_button=False, elem_id="logo-img")
    gr.HTML("""
    <style>
      #logo-img img { height:90px; margin:10px; border-radius:6px; }
    </style>
    """)
    # title & share link
    gr.Markdown("## 🌲 Spatchat: Landscape Metrics Assistant")
    gr.HTML('''
      <div style="margin:-10px 0 20px;">
        <input id="shareLink" type="text" readonly value="https://spatchat.org/browse/?room=landmetrics"
               style="width:50%;padding:5px;border:1px solid #ccc;border-radius:4px;" />
        <button onclick="navigator.clipboard.writeText(document.getElementById('shareLink').value)"
                style="padding:5px 10px;background:#007BFF;color:#fff;border:none;border-radius:4px;">
          📋 Copy Share Link
        </button>
      </div>
    ''')

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload GeoTIFF", type="filepath")
            raster_output = gr.Plot(label="Raster Preview", visible=True)
            clear_raster_button = gr.Button("Clear Raster")
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(value=initial_history, type="messages",
                                 label="Spatchat Dialog")
            question_input = gr.Textbox(placeholder="e.g. Calculate edge_density", lines=1)
            clear_button = gr.Button("Clear Chat")

    # callbacks
    file_input.change(preview_raster, inputs=file_input, outputs=raster_output)
    file_input.change(lambda h=initial_history: initial_history,
                      inputs=None, outputs=chatbot)
    clear_raster_button.click(clear_raster, inputs=None,
                              outputs=[file_input, raster_output])
    question_input.submit(analyze_raster,
                          inputs=[file_input, question_input, chatbot],
                          outputs=[chatbot, question_input])
    clear_button.click(lambda: initial_history, outputs=chatbot)

iface.launch()
