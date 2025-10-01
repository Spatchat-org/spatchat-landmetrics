# metrics_catalog.py
# Metric glossary, synonyms, reverse map; method maps; partitions; list_metrics_text()

metric_definitions = {
    # Patch-level
    "area":  ("Patch Area (AREA)", "Area of each patch"),
    "perim": ("Patch Perimeter (PERIM)", "Perimeter of each patch"),
    "para":  ("Perimeter–Area Ratio (PARA)", "p/a per patch"),
    "shape": ("Shape Index (SHAPE)", "Normalized shape per patch"),
    "frac":  ("Fractal Dimension (FRAC)", "Fractal complexity per patch"),
    "enn":   ("Euclidean Nearest Neighbor (ENN)", "Nearest same-class patch"),
    "core_p":("Core Area (CORE)", "Core area per patch"),
    "ncore": ("Number of Core Areas (NCORE)", "Disjunct cores per patch"),
    "cai":   ("Core Area Index (CAI)", "CORE / AREA per patch"),

    # Class & Landscape (area/edge)
    "ta":    ("Total Area (TA)", "Total area (class/landscape)"),
    "pland": ("Proportion of Landscape (PLAND)", "% of landscape by class"),
    "np":    ("Number of Patches (NP)", "Total patches"),
    "pd":    ("Patch Density (PD)", "Patches per area"),
    "lpi":   ("Largest Patch Index (LPI)", "% landscape by largest patch"),
    "te":    ("Total Edge (TE)", "Total edge length"),
    "edge_density": ("Edge Density (ED)", "Edge length per area unit"),

    # Core family
    "tca":   ("Total Core Area (TCA)", "Sum of core areas"),
    "cpland":("Core Area % of Landscape (CPLAND)", "% core area of landscape (per class)"),
    "ndca":  ("Number of Disjunct Core Areas (NDCA)", "Count of disjunct cores"),
    "dcad":  ("Disjunct Core Area Density (DCAD)", "Core areas per area unit"),

    # Aggregation/connectivity
    "lsi":   ("Landscape Shape Index (LSI)", "Standardized edginess"),
    "mesh":  ("Effective Mesh Size (MESH)", "Aggregation based on patch sizes"),

    # Diversity / contagion (landscape-only)
    "entropy":                    ("Entropy (H)", "Information-theoretic diversity"),
    "joint_entropy":              ("Joint Entropy H(X,Y)", "Adjacency complexity"),
    "conditional_entropy":        ("Conditional Entropy H(Y|X)", "Spatial complexity"),
    "mutual_information":         ("Mutual Information (MI)", "Aggregation"),
    "relative_mutual_information":("Relative Mutual Information (RMI)", "Normalized MI"),
    "shdi":                       ("Shannon Diversity Index (SHDI)", "Diversity"),
    "contag":                     ("Contagion (CONTAG)", "Aggregation of adjacencies"),
}

synonyms = {
    # Patch-level
    "area":  ["area", "patch area", "area_patch", "area (patch)"],
    "perim": ["perimeter", "perim", "patch perimeter"],
    "para":  ["para", "perimeter-area ratio", "perimeter area ratio"],
    "shape": ["shape", "shape index"],
    "frac":  ["frac", "fractal dimension"],
    "enn":   ["enn", "nearest neighbor", "nearest-neighbor"],
    "core_p":["core", "core area", "core (patch)", "patch core"],
    "ncore": ["ncore", "number of core areas", "num core areas"],
    "cai":   ["cai", "core area index"],

    # Class & landscape
    "ta":    ["ta", "total area"],
    "pland": ["pland", "proportion of landscape", "percent landscape", "% landscape",
              "percent of landscape", "percent cover", "percentage landscape", "percent composition"],
    "np":    ["np", "number of patches"],
    "pd":    ["pd", "patch density"],
    "lpi":   ["lpi", "largest patch index"],
    "te":    ["te", "total edge"],
    "edge_density": ["ed", "edge density"],

    # Core family
    "tca":   ["tca", "total core area"],
    "cpland":["cpland", "core area proportion of landscape", "core area percent of landscape"],
    "ndca":  ["ndca", "number of disjunct core areas"],
    "dcad":  ["dcad", "disjunct core area density"],

    # Aggregation/connectivity
    "lsi":   ["lsi", "landscape shape index"],
    "mesh":  ["mesh", "effective mesh size"],

    # Diversity/contagion
    "entropy":                    ["entropy", "h"],
    "joint_entropy":              ["joint entropy", "h(x,y)", "hxy", "joinent"],
    "conditional_entropy":        ["conditional entropy", "h(y|x)", "hyx", "condent"],
    "mutual_information":         ["mutual information", "mi", "mutinf"],
    "relative_mutual_information":["relative mutual information", "rmi", "relmutinf"],
    "shdi":                       ["shdi", "shannon diversity"],
    "contag":                     ["contag", "contagion"],
}

# Reverse synonym map
reverse_synonyms = {}
for code, syn_list in synonyms.items():
    for syn in syn_list:
        reverse_synonyms[syn.lower()] = code

# Class/Landscape methods
metric_map = {
    "ta":           "total_area",
    "pland":        "proportion_of_landscape",
    "np":           "number_of_patches",
    "pd":           "patch_density",
    "lpi":          "largest_patch_index",
    "te":           "total_edge",
    "edge_density": "edge_density",
    "tca":          "total_core_area",
    "cpland":       "core_area_proportion_of_landscape",
    "ndca":         "number_of_disjunct_core_areas",
    "dcad":         "disjunct_core_area_density",
    "lsi":          "landscape_shape_index",
    "mesh":         "effective_mesh_size",

    # Landscape-only
    "entropy":                    "entropy",
    "joint_entropy":              "joint_entropy",
    "conditional_entropy":        "conditional_entropy",
    "mutual_information":         "mutual_information",
    "relative_mutual_information":"relative_mutual_information",
    "shdi":                       "shannon_diversity_index",
    "contag":                     "contagion",
}

# Patch-level methods
patch_metric_map = {
    "area":  "area",
    "perim": "perimeter",
    "para":  "perimeter_area_ratio",
    "shape": "shape_index",
    "frac":  "fractal_dimension",
    "enn":   "euclidean_nearest_neighbor",
    "core_p":"core_area",
    "ncore": "number_of_core_areas",
    "cai":   "core_area_index",
}

# Helper and partitions
helper_methods = metric_map.copy()

patch_only = set(patch_metric_map.keys())
class_only = {"pland", "cpland"}
cross_level = {"ta","np","pd","lpi","te","edge_density","tca","ndca","dcad","lsi","mesh"}
landscape_only = {"entropy","joint_entropy","conditional_entropy",
                  "mutual_information","relative_mutual_information","shdi","contag"}

def list_metrics_text():
    lines = []
    lines.append("**Patch-level:**")
    lines += [f"- {metric_definitions[k][0]} (`{k}`)" for k in sorted(patch_only)]

    lines.append("\n**Class-level:**")
    lines += [f"- {metric_definitions[k][0]} (`{k}`)" for k in sorted(class_only)]

    lines.append("\n**Cross-level (class & landscape):**")
    lines += [f"- {metric_definitions[k][0]} (`{k}`)" for k in sorted(cross_level)]

    lines.append("\n**Landscape-only:**")
    lines += [f"- {metric_definitions[k][0]} (`{k}`)" for k in sorted(landscape_only)]
    return "\n".join(lines)
