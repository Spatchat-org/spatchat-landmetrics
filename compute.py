# compute.py
from metrics_catalog import (
    metric_definitions, metric_map, patch_metric_map, helper_methods,
    patch_only, class_only, cross_level, landscape_only, list_metrics_text
)
from raster_io import _build_landscape, answer_metadata, count_classes

def compute_landscape_only_text(file, keys):
    ls = _build_landscape(file)
    parts = ["**Landscape-level metrics:**"]
    for key in keys:
        name,_ = metric_definitions[key]
        if key == "np":
            df = ls.compute_class_metrics_df(metrics=["number_of_patches"])
            val = int(df["number_of_patches"].sum())
        elif key == "cpland":
            df = ls.compute_class_metrics_df(metrics=["total_core_area","total_area"])
            tca_sum = float(df["total_core_area"].sum())
            ta_sum  = float(df["total_area"].sum())
            val = 0.0 if ta_sum == 0 else (100.0 * tca_sum / ta_sum)
        elif key in helper_methods:
            val = getattr(ls, helper_methods[key])()
        else:
            df  = ls.compute_landscape_metrics_df(metrics=[metric_map[key]])
            val = df[metric_map[key]].iloc[0]
        parts.append(
            f"**{name} ({key.upper()}):** {val:.4f}"
            if isinstance(val, float) else f"**{name} ({key.upper()}):** {val}"
        )
    return "\n\n".join(parts)

def compute_class_only_text(file, keys):
    ls   = _build_landscape(file)
    cols = [metric_map[k] for k in keys]
    df   = ls.compute_class_metrics_df(metrics=cols)
    if "class_val" not in df.columns:
        df = df.rename_axis("class_val").reset_index()
    else:
        df = df.reset_index(drop=True)
    df["class_name"] = df["class_val"].astype(int).apply(lambda c: f"Class {c}")
    display_cols = ["class_name"] + [metric_map[k] for k in keys]
    tbl  = df[display_cols].to_markdown(index=False)
    return f"**Class-level metrics:**\n{tbl}"

def compute_patch_only_text(file, keys, row_cap=50):
    ls   = _build_landscape(file)
    cols = [patch_metric_map[k] for k in keys]
    df   = ls.compute_patch_metrics_df(metrics=cols).reset_index()
    total_rows = len(df)
    if total_rows > row_cap:
        df = df.head(row_cap)
        note = f"\n\n_Showing first {row_cap} of {total_rows} patches._"
    else:
        note = ""
    tbl  = df.to_markdown(index=False)
    return f"**Patch-level metrics:**\n{tbl}{note}"

def compute_multiple_metrics_text(file, keys):
    land_keys  = [k for k in keys if (k in cross_level or k in landscape_only)]
    class_keys = [k for k in keys if k in class_only]
    patch_keys = [k for k in keys if k in patch_only]
    parts = []
    if land_keys:
        parts.append(compute_landscape_only_text(file, land_keys))
    if class_keys:
        parts.append(compute_class_only_text(file, class_keys))
    if patch_keys:
        parts.append(compute_patch_only_text(file, patch_keys))
    return "\n\n".join(parts)

# Tool wrappers expected by the handler
def run_metadata(file):
    return answer_metadata(file)

def run_count_classes(file):
    return count_classes(file)

def run_list_metrics(file):
    return list_metrics_text()

def run_compute_metrics(file, raw_metrics, level):
    from metrics_catalog import reverse_synonyms  # local import avoids cycles
    mapped, unknown = [], []
    for m in raw_metrics:
        ml = str(m).lower()
        if ml in metric_definitions:
            cand = ml
        elif ml in reverse_synonyms:
            cand = reverse_synonyms[ml]
        else:
            unknown.append(m); continue
        if level == "patch" and cand == "core":
            cand = "core_p"
        if (cand in metric_map) or (cand in patch_metric_map):
            mapped.append(cand)
        else:
            unknown.append(m)

    if unknown:
        return f"Sorry, I don’t recognize: {', '.join(map(str, unknown))}. Could you clarify?"

    if level == "patch":
        keys = [k for k in mapped if k in patch_only]
        return compute_patch_only_text(file, keys) if keys else "No patch-level metrics requested."

    land = [c for c in mapped if (c in cross_level or c in landscape_only)]
    clas = [c for c in mapped if c in class_only]
    any_cross = bool(set(mapped) & cross_level)

    if level == "class":
        class_metrics = [m for m in mapped if m in (class_only | cross_level)]
        return compute_class_only_text(file, class_metrics)

    if level == "landscape":
        if not land and clas:
            return compute_class_only_text(file, clas)
        return compute_landscape_only_text(file, land or [m for m in mapped if m in (cross_level | landscape_only)])

    if any_cross:
        return compute_multiple_metrics_text(file, mapped)
    return compute_class_only_text(file, clas or mapped)
