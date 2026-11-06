# raster_io.py
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gradio as gr
from pylandstats import Landscape

def _filepath(file):
    # Accept either gradio file object with .name or plain string path
    return getattr(file, "name", file)

def no_raster_fig():
    fig, ax = plt.subplots(figsize=(5,5))
    ax.text(0.5,0.5,"No raster loaded.\nUpload or drag a GeoTIFF here,\nor click 'Try sample data'.",
            ha='center',va='center',color='gray',fontsize=12)
    ax.set_title("Raster Preview", color='dimgray')
    ax.axis('off')
    return fig

def preview_raster(file):
    if not file:
        return no_raster_fig()
    path = _filepath(file)
    with rasterio.open(path) as src:
        raw = src.read(1); nodata = src.nodata or 0
    if raw.dtype.kind in ('U','S','O'):
        data = raw.astype(str)
        uniq = np.unique(data[data!=""])
        arr  = np.zeros_like(raw,dtype=int)
        for i,nm in enumerate(uniq,1):
            arr[data==nm] = i
        labels = [f"{i}: {nm}" for i,nm in enumerate(uniq,1)]
    else:
        arr = raw
        vals = np.unique(arr[arr!=nodata])
        labels = [f"Class {int(v)}" for v in vals]
    n = len(labels)
    colors = plt.cm.tab10(np.linspace(0,1,n))
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(arr,cmap='tab10'); ax.set_title("Uploaded Raster"); ax.axis('off')
    handles = [mpatches.Patch(color=c,label=l) for c,l in zip(colors,labels)]
    ax.legend(handles=handles,loc='lower left',fontsize='small')
    return fig

def clear_raster():
    return None, gr.update(value=no_raster_fig(), visible=True)

def notify_upload(file, history):
    if file:
        return history + [{"role":"assistant","content":"📥 Raster uploaded successfully!"}], ""
    return history, ""

def answer_metadata(file):
    path = _filepath(file)
    with rasterio.open(path) as src:
        return (
            f"CRS: {src.crs}\n"
            f"Resolution: {src.res[0]:.2f}×{src.res[1]:.2f}\n"
            f"Extent: {src.bounds}\n"
            f"Bands: {src.count}\n"
            f"NoData: {src.nodata}"
        )

def count_classes(file):
    path = _filepath(file)
    with rasterio.open(path) as src:
        arr = src.read(1); nodata = src.nodata or 0
    vals = np.unique(arr[arr!=nodata])
    return f"Your raster contains {len(vals)} unique classes."

def _build_landscape(file):
    path = _filepath(file)
    with rasterio.open(path) as src:
        raw = src.read(1)
        x_res,y_res = src.res
        nodata = src.nodata or 0
    if raw.dtype.kind in ("U","S","O"):
        data_str   = raw.astype(str)
        uniq_names = np.unique(data_str[data_str!=""])
        name2code  = {nm:i+1 for i,nm in enumerate(uniq_names)}
        arr        = np.zeros_like(raw,dtype=int)
        for nm,code in name2code.items():
            arr[data_str==nm] = code
        return Landscape(arr, res=(x_res,y_res), nodata=0)
    else:
        return Landscape(path, nodata=nodata, res=(x_res,y_res))
