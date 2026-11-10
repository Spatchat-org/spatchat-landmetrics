# data_views.py
"""What the Datasheet and Plots tabs show for an uploaded land-cover raster."""
import os
import tempfile
import time
import uuid

import numpy as np
import pandas as pd
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import ndimage  # noqa: E402

import spatchat_layout  # noqa: E402


def _filepath(file):
    return getattr(file, "name", file)


def _load_classes(file):
    """Return (int class array, class labels dict, (x_res, y_res), nodata mask)."""
    path = _filepath(file)
    with rasterio.open(path) as src:
        raw = src.read(1)
        x_res, y_res = src.res
        nodata = src.nodata if src.nodata is not None else 0
    if raw.dtype.kind in ("U", "S", "O"):
        data = raw.astype(str)
        names = np.unique(data[data != ""])
        arr = np.zeros(raw.shape, dtype=int)
        labels = {}
        for i, nm in enumerate(names, 1):
            arr[data == nm] = i
            labels[i] = str(nm)
        mask = arr == 0
    else:
        arr = raw
        mask = arr == nodata
        if np.issubdtype(arr.dtype, np.floating):
            mask |= np.isnan(arr)
        labels = {int(v): f"Class {int(v)}" for v in np.unique(arr[~mask])}
    return arr, labels, (abs(x_res), abs(y_res)), mask


def class_table(file) -> pd.DataFrame:
    """One row per land-cover class: cells, area, share of the landscape and number of patches."""
    arr, labels, (xr, yr), mask = _load_classes(file)
    valid = int((~mask).sum())
    rows = []
    for code, label in labels.items():
        sel = (arr == code) & ~mask
        n = int(sel.sum())
        patches = int(ndimage.label(sel)[1]) if n else 0
        rows.append({
            "class": label, "value": code, "cells": n,
            "area": round(n * xr * yr, 3),
            "pland_%": round(100.0 * n / valid, 3) if valid else 0.0,
            "patches": patches,
        })
    return pd.DataFrame(rows).sort_values("cells", ascending=False).reset_index(drop=True)


def _out_dir():
    d = os.path.join(tempfile.gettempdir(), "spatchat_landmetrics", uuid.uuid4().hex)
    os.makedirs(d, exist_ok=True)
    return d


def raster_plots(file, out_dir=None) -> list:
    """Class composition bar chart and patch-size distribution (plus the class table as CSV).
    Returns the PNG paths."""
    arr, labels, (xr, yr), mask = _load_classes(file)
    table = class_table(file)
    out = out_dir or _out_dir()
    os.makedirs(out, exist_ok=True)
    table.to_csv(os.path.join(out, "class_summary.csv"), index=False)
    paths = []

    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(table), 1)))
    ax.bar(table["class"].astype(str), table["pland_%"], color=colors)
    ax.set_ylabel("PLAND (% of landscape)")
    ax.set_title("Class composition")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    p = os.path.join(out, "class_composition.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    paths.append(p)

    sizes = []
    for code in labels:
        lab, n = ndimage.label((arr == code) & ~mask)
        if n:
            sizes.extend((np.bincount(lab.ravel())[1:] * xr * yr).tolist())
    if sizes:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.hist(sizes, bins=min(40, max(5, len(sizes) // 3)), color="#fa6a98", edgecolor="white")
        if max(sizes) / max(min(sizes), 1e-9) > 50:
            ax.set_xscale("log")
        ax.set_xlabel("Patch area (map units²)")
        ax.set_ylabel("Number of patches")
        ax.set_title("Patch size distribution")
        fig.tight_layout()
        p = os.path.join(out, "patch_sizes.png")
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    # the table itself, as a figure like the charts (written last so it is the newest and shows first)
    paths.append(spatchat_layout.table_png(table, os.path.join(out, "class_summary_table.png"), "Land-cover classes"))
    return paths


def _parse_markdown_table(text: str):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 3:
        return None
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) == len(header):
            rows.append(cells)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=header)
    for c in df.columns:
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().all():
            df[c] = conv
    return df


def metric_plots_from_reply(text: str, out_dir=None) -> list:
    """Charts for a metrics table in a chat reply: bars per class, histograms for per-patch tables.
    The table itself is saved as a CSV next to the charts."""
    df = _parse_markdown_table(text)
    if df is None or df.empty:
        return []
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in ("class_val", "patch_id", "index")]
    if not num:
        return []
    label_col = next((c for c in df.columns if c not in num), None)
    out = out_dir or _out_dir()
    os.makedirs(out, exist_ok=True)
    stamp = time.strftime("%H%M%S")
    df.to_csv(os.path.join(out, f"metrics_{stamp}.csv"), index=False)
    paths = []
    for col in num[:4]:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        if label_col is not None and len(df) <= 40:
            ax.bar(df[label_col].astype(str), df[col], color="#fa6a98")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
            ax.set_ylabel(col)
            ax.set_title(f"{col} by {label_col}")
        else:
            ax.hist(df[col].dropna(), bins=min(30, max(5, len(df) // 3)), color="#fa6a98", edgecolor="white")
            ax.set_xlabel(col)
            ax.set_ylabel("Number of patches")
            ax.set_title(f"Distribution of {col}")
        fig.tight_layout()
        p = os.path.join(out, f"metric_{stamp}_{col}.png".replace("/", "_"))
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    paths.append(spatchat_layout.table_png(df, os.path.join(out, f"metrics_{stamp}_table.png"), "Landscape metrics"))
    return paths
