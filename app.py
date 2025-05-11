import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import rasterio
import numpy as np
from pylandstats import Landscape
import re
import pandas as pd
import yaml
from dotenv import load_dotenv
import os
from together import Together

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
        handles = [mpatches.Patch(color=colors[i], label=f'Class {int(val)}')
                   for i, val in enumerate(unique)]
        ax.legend(handles=handles, loc='lower left', fontsize='small', frameon=True)
        return fig
    except Exception:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, "🗂️ No raster loaded.", ha='center', va='center', color='gray')
        ax.set_title("Raster Preview")
        ax.axis('off')
        return fig

# --- Clear raster ---
def clear_raster():
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.text(0.5, 0.5, "🗂️ No raster loaded.", ha='center', va='center', color='gray')
    ax.set_title("Raster Preview")
    ax.axis('off')
    return None, gr.update(value=fig)

# --- Load glossary and synonyms ---
glossary = None
for fname in ['spatchat_glossary.yaml', 'spatchat_glossary.yml', 'spatchat_glossary.txt']:
    try:
        with open(fname) as f:
            glossary = yaml.safe_load(f)
        if not isinstance(glossary, dict):
            raise ValueError("Glossary file did not contain a mapping")
        break
    except FileNotFoundError:
        continue
    except Exception as e:
        print(f"⚠️ Warning loading glossary from {fname}: {e}")
        break
if glossary is None:
    glossary = {
        "pland": {"name": "Proportion of Landscape (PLAND)", "definition": "% of landscape by class", "units": "%", "interpretation": "..."},
        "np": {"name": "Number of Patches (NP)", "definition": "Count of patches", "units": "unitless", "interpretation": "..."},
    }
metric_definitions = {code: info for code, info in glossary.items()}
synonyms = {}
for code, info in glossary.items():
    nm = info.get('name', code).lower()
    phrase = code.replace('_', ' ')
    synonyms[code] = [code.lower(), phrase, nm]

# --- LLM setup ---
load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

# --- Helpers ---
def answer_metadata(src, query_lower):
    parts = []
    x_res, y_res = src.res
    unit = getattr(src.crs, 'linear_units', 'unit')
    if 'resolution' in query_lower:
        parts.append(f"Resolution: {x_res:.2f} × {y_res:.2f} {unit}")
    if 'crs' in query_lower:
        parts.append(f"CRS: {src.crs}")
    if 'extent' in query_lower:
        parts.append(f"Extent: {src.bounds}")
    if 'bands' in query_lower:
        parts.append(f"Bands: {src.count}")
    if 'nodata' in query_lower:
        parts.append(f"NoData: {src.nodata}")
    return "    ".join(parts)


def list_metrics():
    categories = {
        "Landscape": ["contag", "shdi", "shei", "mesh", "lsi", "tca"],
        "Class": ["pland", "np", "pd", "lpi", "total_edge", "edge_density"],
        "Patch": ["area", "perim", "para", "shape", "frac", "enn", "core", "nca", "cai"]
    }
    lines = []
    for level, keys in categories.items():
        lines.append(f"**{level}-level metrics:**")
        for k in keys:
            name = glossary[k]['name']
            definition = glossary[k]['definition']
            lines.append(f"- {name} (`{k}`): {definition}")
    return "\n".join(lines)


def compute_metric(file, key):
    landscape = Landscape(file.name, nodata=0)
    df_land = landscape.compute_landscape_metrics_df()
    df_class = landscape.compute_class_metrics_df()

    with rasterio.open(file.name) as src:
        crs = src.crs
        x_res, y_res = src.res
        extent = src.bounds
        bands = src.count
        nodata = src.nodata

    meta = (
        f"CRS: {crs}\n"
        f"Resolution: {x_res:.2f}×{y_res:.2f}\n"
        f"Extent: {extent}\n"
        f"Bands: {bands}\n"
        f"NoData: {nodata}\n\n"
    )

    name = glossary[key]['name']
    landscape_part = ""
    if key in df_land.columns:
        val = df_land[key].iloc[0]
        landscape_part = f"**Landscape-level {name}:** {val:.4f}\n\n"

    dfc = df_class.rename_axis('code').reset_index()
    dfc['class_name'] = dfc['code'].map(lambda c: f"Class {int(c)}")
    table = dfc[['class_name', key]].to_markdown(index=False)

    return meta + landscape_part + f"**Class-level {name}:**\n{table}"


# --- Analyzer with intent dispatch ---
def analyze_raster(file, question, history):
    history = history + [{"role": "user", "content": question}]
    lower_q = question.lower()

    # Intent detection via simple keywords (no separate LLM pass)
    if file is None:
        return history + [{"role": "assistant", "content": "Please upload a GeoTIFF before asking anything."}], ""

    # Metadata intent
    if re.search(r"\b(resolution|crs|extent|bands|nodata)\b", lower_q):
        with rasterio.open(file.name) as src:
            resp = answer_metadata(src, lower_q)
        return history + [{"role": "assistant", "content": resp}], ""

    # List metrics intent
    if re.search(r"\b(what|which|list|available).*metrics\b", lower_q):
        return history + [{"role": "assistant", "content": list_metrics()}], ""

    # Compute metric intent
    matched = None
    for key, syns in synonyms.items():
        if any(re.search(rf"\b{re.escape(s)}\b", lower_q) for s in syns):
            matched = key
            break
    if matched:
        resp = compute_metric(file, matched)
        return history + [{"role": "assistant", "content": resp}], ""

    # Explain metric intent
    if re.search(r"\b(explain|what does).*(metric|index)\b", lower_q):
        key = next((k for k in glossary if any(re.search(rf"\b{re.escape(s)}\b", lower_q) for s in synonyms[k])), None)
        if key:
            info = glossary[key]
            text = (
                f"**{info['name']}**\n{info['definition']}\n"
                f"Units: {info['units']}\n"
                f"Interpretation: {info['interpretation']}"
            )
            return history + [{"role": "assistant", "content": text}], ""
        return history + [{"role": "assistant", "content": "Which metric would you like explained?"}], ""

    # Fallback: LLM casual response
    system_prompt = (
        "You are Spatchat, a helpful assistant.\n"
        "Use rasterio to describe raster metadata when requested.\n"
        "Use pylandstats to calculate metrics when requested.\n"
        "Refer to the glossary for definitions, units, and interpretations.\n"
        "If level isn't specified, return both class and landscape levels.\n"
        "For other queries, reply conversationally without code examples unless asked."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": "A raster file is available for analysis."},
        *history
    ]
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages,
        temperature=0.4
    ).choices[0].message.content
    return history + [{"role": "assistant", "content": response}], ""

# --- UI layout ---
initial_history = [{"role": "assistant", "content": "👋 Hi! I’m Spatchat. Upload a GeoTIFF to begin."}]
with gr.Blocks(title="Spatchat") as iface:
    gr.HTML('<head><link rel="icon" href="file=logo1.png"></head>')
    gr.Image(value="logo/logo_long1.png", show_label=False, type="filepath", elem_id="logo-img")
    gr.HTML('<style>#logo-img img { height:90px; margin:10px;border-radius:6px; }</style>')
    gr.Markdown("## 🌲 Spatchat: Landscape Metrics Assistant")
    gr.HTML('''
    <div style="margin:-10px 0 15px;">
      <input id="shareLink" value="https://spatchat.org/browse/?room=landmetrics" readonly style="width:50%;padding:5px;border:1px solid #ccc;border-radius:4px;">
      <button onclick="navigator.clipboard.writeText(document.getElementById('shareLink').value)" style="padding:5px 10px;background:#007BFF;color:#fff;border:none;border-radius:4px;">📋 Copy Link</button>
      <span style="margin-left:10px;font-size:14px;">Share: <a href="https://twitter.com/intent/tweet?url=https://spatchat.org/browse/?room=landmetrics">🐦</a> | <a href="https://facebook.com/sharer/sharer.php?u=https://spatchat.org/browse/?room=landmetrics">📘</a></span>
    </div>
    ''')
    gr.Markdown('<div style="font-size:14px;">© 2025 Ho Yi Wan & Logan Hysen</div>')
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Upload GeoTIFF", type="filepath")
            raster_output = gr.Plot(label="Raster Preview")
            clear_btn = gr.Button("Clear Raster")
        with gr.Column(scale=1):
            chatbot = gr.Chatbot(value=initial_history, type="messages", label="Spatchat Dialog")
            question = gr.Textbox(label="Ask Spatchat", placeholder="e.g., calculate edge density?", lines=1)
            clear_chat = gr.Button("Clear Chat")
    file_input.change(preview_raster, inputs=file_input, outputs=raster_output)
    clear_btn.click(clear_raster, inputs=None, outputs=[file_input, raster_output])
    question.submit(analyze_raster, inputs=[file_input, question, chatbot], outputs=[chatbot, question])
    clear_chat.click(lambda: initial_history, inputs=None, outputs=chatbot)
iface.launch()
