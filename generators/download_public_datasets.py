from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import json
import math
import random
import re
import shutil
import statistics
import urllib.request
from collections import defaultdict, deque
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets"
SOURCES_DIR = DATA_DIR / "_sources"

BNLEARN_BASE = "https://www.bnlearn.com/bnrepository"
GRAPH_DEFS_URL = (
    "https://raw.githubusercontent.com/AniketVashishtha/"
    "Causal_Order_Imperfect_Experts/main/causal_discovery/graphs/definitions.py"
)
OBSERVABLE_NEUROPATHIC_URL = (
    "https://api.observablehq.com/@turuibo/"
    "the-complete-causal-graph-of-neuropathic-pain-diagnosis.js?v=3"
)

SAMPLE_SIZES = [250, 500, 1000, 5000, 10000]
GRAPH_ONLY_SAMPLE_SIZES = [1000, 5000]
RNG_SEED = 20260601


BNLEARN_DATASETS = {
    "asia": {
        "title": "ASIA / Lung Cancer",
        "nodes": 8,
        "arcs": 8,
        "parameters": 18,
        "context": (
            "Respiratory diagnosis network for a patient who recently visited "
            "Asia; used by Vashishtha et al. and Ban et al."
        ),
        "reference": (
            "S. Lauritzen and D. Spiegelhalter, Local Computation with "
            "Probabilities on Graphical Structures and their Application to "
            "Expert Systems, 1988."
        ),
    },
    "cancer": {
        "title": "CANCER",
        "nodes": 5,
        "arcs": 4,
        "parameters": 10,
        "context": (
            "Small cancer/outcome Bayesian network; heavily used for LLM "
            "pairwise and causal-order examples."
        ),
        "reference": (
            "K. B. Korb and A. E. Nicholson, Bayesian Artificial Intelligence, "
            "2nd edition, Section 2.2.2, 2010."
        ),
    },
    "earthquake": {
        "title": "EARTHQUAKE",
        "nodes": 5,
        "arcs": 4,
        "parameters": 10,
        "context": (
            "Burglary/earthquake/alarm network used in the Vashishtha "
            "causal-order experiments."
        ),
        "reference": (
            "K. B. Korb and A. E. Nicholson, Bayesian Artificial Intelligence, "
            "2nd edition, Section 2.5.1, 2010."
        ),
    },
    "survey": {
        "title": "SURVEY",
        "nodes": 6,
        "arcs": 6,
        "parameters": 21,
        "context": (
            "Hypothetical transport survey network used in Vashishtha et al."
        ),
        "reference": (
            "M. Scutari and J.-B. Denis, Bayesian Networks: with Examples in R, "
            "2nd edition, 2021."
        ),
    },
    "sachs": {
        "title": "SACHS",
        "nodes": 11,
        "arcs": 17,
        "parameters": 178,
        "context": (
            "Protein-signaling Bayesian network included as a useful "
            "supplementary biological benchmark."
        ),
        "reference": (
            "K. Sachs et al., Causal Protein-Signaling Networks Derived from "
            "Multiparameter Single-Cell Data, Science, 2005."
        ),
    },
    "child": {
        "title": "CHILD",
        "nodes": 20,
        "arcs": 25,
        "parameters": 230,
        "context": (
            "Congenital heart disease network; medium-sized and important in "
            "both Vashishtha et al. and Ban et al."
        ),
        "reference": (
            "D. J. Spiegelhalter and S. L. Lauritzen, Sequential updating of "
            "conditional probabilities on directed graphical structures, 1990."
        ),
    },
    "insurance": {
        "title": "INSURANCE",
        "nodes": 27,
        "arcs": 52,
        "parameters": 1008,
        "context": (
            "Car-insurance claim-cost network used by Ban et al.; good "
            "medium-density benchmark for LLM priors."
        ),
        "reference": (
            "J. Binder, D. Koller, S. Russell and K. Kanazawa, Adaptive "
            "probabilistic networks with hidden variables, 1997."
        ),
    },
    "alarm": {
        "title": "ALARM",
        "nodes": 37,
        "arcs": 46,
        "parameters": 509,
        "context": (
            "A Logical Alarm Reduction Mechanism for patient monitoring; used "
            "by Ban et al. as a medium-sized medical-style benchmark."
        ),
        "reference": (
            "I. Beinlich et al., The ALARM Monitoring System: A Case Study "
            "with Two Probabilistic Inference Techniques for Belief Networks, "
            "1989."
        ),
    },
    "mildew": {
        "title": "MILDEW",
        "nodes": 35,
        "arcs": 46,
        "parameters": 540150,
        "context": (
            "Large-parameter agricultural network used by Ban et al.; useful "
            "for testing robustness to weaker LLM priors."
        ),
        "reference": (
            "K. G. Olesen et al., A MUNIN network for diagnosis and "
            "treatment of mildew in winter wheat, 1989."
        ),
    },
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download(url: str, out_path: Path) -> None:
    ensure_dir(out_path.parent)
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36"
            )
        },
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        with out_path.open("wb") as f:
            shutil.copyfileobj(response, f)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ungzip(gz_path: Path, out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    with gzip.open(gz_path, "rb") as src, out_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"//.*?$", "", text, flags=re.M)
    return text


def parse_bif(path: Path) -> dict:
    text = strip_comments(path.read_text(encoding="utf-8"))
    variables = {}
    var_re = re.compile(
        r"variable\s+([^\s{]+)\s*\{\s*type\s+discrete\s*"
        r"\[\s*(\d+)\s*\]\s*\{(.*?)\}\s*;\s*\}",
        flags=re.S,
    )
    for m in var_re.finditer(text):
        name = m.group(1)
        states = [s.strip().strip('"') for s in m.group(3).split(",")]
        variables[name] = {"states": states, "state_count": int(m.group(2))}

    cpts = {}
    prob_re = re.compile(
        r"probability\s*\(\s*([^|\)]+?)\s*(?:\|\s*([^\)]+?))?\s*\)\s*\{(.*?)\}",
        flags=re.S,
    )
    for m in prob_re.finditer(text):
        child = m.group(1).strip()
        parents = []
        if m.group(2):
            parents = [p.strip() for p in m.group(2).split(",")]
        body = m.group(3)
        rows = {}
        table = re.search(r"table\s+([^;]+);", body, re.S)
        if table:
            rows[""] = [float(x.strip()) for x in table.group(1).replace("\n", " ").split(",")]
        else:
            for rm in re.finditer(r"\((.*?)\)\s*([^;]+);", body, flags=re.S):
                key = tuple(v.strip().strip('"') for v in rm.group(1).split(","))
                probs = [float(x.strip()) for x in rm.group(2).replace("\n", " ").split(",")]
                rows["|".join(key)] = probs
        cpts[child] = {"parents": parents, "rows": rows}

    for name in variables:
        if name not in cpts:
            raise ValueError(f"Missing CPT for {name} in {path}")
    return {"variables": variables, "cpts": cpts}


def graph_edges(parsed_bif: dict) -> list[tuple[str, str]]:
    edges = []
    for child, spec in parsed_bif["cpts"].items():
        for parent in spec["parents"]:
            edges.append((parent, child))
    return edges


def topological_order(nodes: list[str], edges: list[tuple[str, str]]) -> list[str]:
    children = defaultdict(list)
    indegree = {n: 0 for n in nodes}
    for src, dst in edges:
        if src not in indegree:
            indegree[src] = 0
        if dst not in indegree:
            indegree[dst] = 0
        children[src].append(dst)
        indegree[dst] += 1
    q = deque([n for n in nodes if indegree.get(n, 0) == 0])
    order = []
    while q:
        node = q.popleft()
        order.append(node)
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                q.append(child)
    if len(order) != len(indegree):
        raise ValueError("Graph contains a cycle or unresolved node names.")
    return order


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sample_bn(parsed_bif: dict, n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    nodes = list(parsed_bif["variables"].keys())
    order = topological_order(nodes, graph_edges(parsed_bif))
    data = []
    for _ in range(n):
        row = {}
        for node in order:
            states = parsed_bif["variables"][node]["states"]
            spec = parsed_bif["cpts"][node]
            if spec["parents"]:
                key = "|".join(row[p] for p in spec["parents"])
            else:
                key = ""
            probs = spec["rows"][key]
            row[node] = rng.choices(states, weights=probs, k=1)[0]
        data.append({node: row[node] for node in nodes})
    return data


def read_graph_definitions() -> dict:
    defs_path = SOURCES_DIR / "Causal_Order_Imperfect_Experts" / "causal_discovery" / "graphs" / "definitions.py"
    if not defs_path.exists():
        defs_path = SOURCES_DIR / "causal_order_definitions.py"
        download(GRAPH_DEFS_URL, defs_path)

    tree = ast.parse(defs_path.read_text(encoding="utf-8"))
    env = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            value = ast.literal_eval(node.value) if _is_literal(node.value) else None
            if value is not None:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        env[target.id] = value
    graphs = env.get("GRAPHS")
    if not graphs:
        graphs = {
            k.lower(): v
            for k, v in env.items()
            if isinstance(v, dict) and {"nodes", "ground_truth_edges"} <= set(v)
        }
    return graphs


def _is_literal(node: ast.AST) -> bool:
    try:
        ast.literal_eval(node)
        return True
    except Exception:
        return False


def clean_name(name: str) -> str:
    return " ".join(str(name).strip().split())


def normalize_graph(raw_graph: dict, include_missing_edge_endpoints: bool = True) -> dict:
    raw_nodes = [clean_name(n) for n in raw_graph["nodes"]]
    descriptions = raw_graph.get("descriptions") or {}
    clean_desc = {clean_name(k): v for k, v in descriptions.items()} if descriptions else {}

    nodes = []
    for node in raw_nodes:
        if node not in nodes:
            nodes.append(node)
    canonical_by_lower = {node.lower(): node for node in nodes}

    def canonical(name: str) -> str:
        cleaned = clean_name(name)
        return canonical_by_lower.get(cleaned.lower(), cleaned)

    edges = []
    raw_edges = []
    for src, dst in raw_graph["ground_truth_edges"]:
        s, d = canonical(src), canonical(dst)
        raw_edges.append((s, d))
        if include_missing_edge_endpoints:
            if s not in nodes:
                nodes.append(s)
                canonical_by_lower[s.lower()] = s
            if d not in nodes:
                nodes.append(d)
                canonical_by_lower[d.lower()] = d
            if (s, d) not in edges:
                edges.append((s, d))
        elif s in nodes and d in nodes and (s, d) not in edges:
            edges.append((s, d))

    missing = sorted({x for e in raw_edges for x in e if x not in raw_nodes})
    return {
        "nodes": nodes,
        "raw_nodes": raw_nodes,
        "edges": edges,
        "raw_edges": raw_edges,
        "missing_edge_endpoints": missing,
        "descriptions": clean_desc,
        "context": raw_graph.get("context", ""),
    }


def synthetic_sem_samples(nodes: list[str], edges: list[tuple[str, str]], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    order = topological_order(nodes, edges)
    parents = defaultdict(list)
    for src, dst in edges:
        parents[dst].append(src)
    weights = {
        (src, dst): rng.uniform(0.35, 1.15) * (-1 if rng.random() < 0.35 else 1)
        for src, dst in edges
    }
    rows = []
    for _ in range(n):
        row = {}
        for node in order:
            val = rng.gauss(0, 1)
            if parents[node]:
                val += sum(weights[(p, node)] * row[p] for p in parents[node])
                # A light nonlinearity keeps the data realistic without hiding the DAG.
                val = math.tanh(val / (1 + 0.2 * len(parents[node]))) + rng.gauss(0, 0.35)
            row[node] = val
        rows.append({node: round(row[node], 6) for node in nodes})
    return rows


def save_bnlearn_dataset(name: str, meta: dict, index_rows: list[dict]) -> None:
    ds_dir = DATA_DIR / "bnlearn" / name
    raw_dir = ds_dir / "raw"
    samples_dir = ds_dir / "samples"
    ensure_dir(raw_dir)
    ensure_dir(samples_dir)

    for ext in ["bif", "dsc", "net"]:
        gz = raw_dir / f"{name}.{ext}.gz"
        download(f"{BNLEARN_BASE}/{name}/{name}.{ext}.gz", gz)
        ungzip(gz, raw_dir / f"{name}.{ext}")

    for ext in ["rda", "rds"]:
        download(f"{BNLEARN_BASE}/{name}/{name}.{ext}", raw_dir / f"{name}.{ext}")

    parsed = parse_bif(raw_dir / f"{name}.bif")
    nodes = list(parsed["variables"].keys())
    edges = graph_edges(parsed)

    write_csv(
        ds_dir / "nodes.csv",
        [
            {
                "node": node,
                "state_count": parsed["variables"][node]["state_count"],
                "states": "|".join(parsed["variables"][node]["states"]),
                "description": "",
            }
            for node in nodes
        ],
        ["node", "state_count", "states", "description"],
    )
    write_csv(
        ds_dir / "edges.csv",
        [{"source": src, "target": dst} for src, dst in edges],
        ["source", "target"],
    )

    (ds_dir / "cpts.json").write_text(json.dumps(parsed["cpts"], indent=2), encoding="utf-8")

    for size in SAMPLE_SIZES:
        out_csv = samples_dir / f"{name}_n{size}.csv"
        if not out_csv.exists():
            rows = sample_bn(parsed, size, RNG_SEED + size + len(name))
            write_csv(out_csv, rows, nodes)

    raw_checksums = {
        p.name: sha256(p)
        for p in sorted(raw_dir.iterdir())
        if p.is_file()
    }
    metadata = {
        "name": name,
        "collection": "bnlearn",
        "title": meta["title"],
        "source": f"{BNLEARN_BASE}/{name}/",
        "source_repository": BNLEARN_BASE + "/",
        "license_note": "bnlearn repository pages are licensed CC BY-SA; check individual data references.",
        "context": meta["context"],
        "reference": meta["reference"],
        "node_count_repository": meta["nodes"],
        "arc_count_repository": meta["arcs"],
        "parameter_count_repository": meta["parameters"],
        "node_count_parsed": len(nodes),
        "edge_count_parsed": len(edges),
        "sample_sizes": SAMPLE_SIZES,
        "csv_generation": "Samples generated from the downloaded BIF CPTs with a fixed seed.",
        "raw_checksums_sha256": raw_checksums,
    }
    (ds_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (ds_dir / "README.md").write_text(render_bnlearn_readme(metadata), encoding="utf-8")

    index_rows.append(
        {
            "collection": "bnlearn",
            "dataset": name,
            "nodes": len(nodes),
            "edges": len(edges),
            "csv_files": len(SAMPLE_SIZES),
            "path": str(ds_dir.relative_to(DATA_DIR)).replace("\\", "/"),
            "notes": "BN CPT samples generated from BIF.",
        }
    )


def render_bnlearn_readme(meta: dict) -> str:
    return f"""# {meta['title']}

Collection: bnlearn Bayesian Network Repository

Source: {meta['source']}

Context: {meta['context']}

Reference: {meta['reference']}

Files:
- raw/: original BIF/DSC/NET/RDA/RDS files downloaded from bnlearn.
- nodes.csv: variables and states parsed from BIF.
- edges.csv: directed ground-truth graph.
- cpts.json: conditional probability tables parsed from BIF.
- samples/: categorical observational CSV samples generated from the BIF CPTs.

Sample sizes: {', '.join(map(str, meta['sample_sizes']))}

Generation note: the CSV files are synthetic observational samples from the public
Bayesian network parameters, not separately published original observations.
"""


def save_graph_only_dataset(name: str, graph: dict, index_rows: list[dict], collection: str = "llm_causal_order_graphs") -> None:
    ds_dir = DATA_DIR / collection / name
    samples_dir = ds_dir / "samples"
    ensure_dir(samples_dir)

    nodes = graph["nodes"]
    edges = graph["edges"]
    descriptions = graph.get("descriptions", {})

    write_csv(
        ds_dir / "nodes.csv",
        [
            {
                "node": node,
                "description": descriptions.get(node, ""),
                "in_raw_node_list": str(node in graph.get("raw_nodes", nodes)).lower(),
            }
            for node in nodes
        ],
        ["node", "description", "in_raw_node_list"],
    )
    write_csv(
        ds_dir / "edges.csv",
        [{"source": src, "target": dst} for src, dst in edges],
        ["source", "target"],
    )
    if graph.get("raw_edges"):
        write_csv(
            ds_dir / "edges_raw.csv",
            [{"source": src, "target": dst} for src, dst in graph["raw_edges"]],
            ["source", "target"],
        )

    for size in GRAPH_ONLY_SAMPLE_SIZES:
        out_csv = samples_dir / f"{name}_synthetic_sem_n{size}.csv"
        rows = synthetic_sem_samples(nodes, edges, size, RNG_SEED + size + len(name))
        write_csv(out_csv, rows, nodes)

    metadata = {
        "name": name,
        "collection": collection,
        "context": graph.get("context", ""),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "source": graph.get("source", "Causal_Order_Imperfect_Experts graph definitions"),
        "missing_edge_endpoints_in_raw_node_list": graph.get("missing_edge_endpoints", []),
        "csv_generation": (
            "No public CPT/observational data is bundled with this graph. "
            "CSV files are generated from a simple continuous SEM for algorithm testing."
        ),
        "sample_sizes": GRAPH_ONLY_SAMPLE_SIZES,
    }
    (ds_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (ds_dir / "README.md").write_text(render_graph_only_readme(metadata), encoding="utf-8")

    index_rows.append(
        {
            "collection": collection,
            "dataset": name,
            "nodes": len(nodes),
            "edges": len(edges),
            "csv_files": len(GRAPH_ONLY_SAMPLE_SIZES),
            "path": str(ds_dir.relative_to(DATA_DIR)).replace("\\", "/"),
            "notes": "Graph-only source; CSV is synthetic SEM data.",
        }
    )


def render_graph_only_readme(meta: dict) -> str:
    warning = ""
    if meta["missing_edge_endpoints_in_raw_node_list"]:
        warning = (
            "\nValidation note: some raw edge endpoints were not present in the "
            "raw node list and were added to nodes.csv so that edges.csv is a "
            "valid DAG. See metadata.json for the exact names.\n"
        )
    return f"""# {meta['name']}

Collection: {meta['collection']}

Source: {meta['source']}

Context: {meta['context']}

Files:
- nodes.csv: node names and descriptions where available.
- edges.csv: directed ground-truth graph used for experiments.
- edges_raw.csv: original edge list before endpoint cleanup, when available.
- samples/: continuous synthetic SEM CSV files generated from the graph.

Important: this graph source does not provide public CPTs or original
observational rows. The CSV files are generated for downstream algorithm
experiments and should be reported as synthetic SEM samples.
{warning}
"""


def extract_observable_neuropathic() -> dict:
    js_path = SOURCES_DIR / "neuropathic_observable.js"
    download(OBSERVABLE_NEUROPATHIC_URL, js_path)
    text = js_path.read_text(encoding="utf-8")
    start = text.index("function _data(){return(") + len("function _data(){return(")
    end = text.index("\n)}\n\nfunction _height", start)
    data = json.loads(text[start:end].strip())

    root = "DLS C6-C7"
    first = [link["target"] for link in data["links"] if link["source"] == root]
    second = [
        link["target"]
        for link in data["links"]
        if link["source"] in {"L C7", "R C7"}
    ]
    nodes = []
    for node in [root] + first + second:
        if node not in nodes:
            nodes.append(node)
    edges = []
    for link in data["links"]:
        edge = (link["source"], link["target"])
        if edge[0] in nodes and edge[1] in nodes and edge not in edges:
            edges.append(edge)
    return {
        "nodes": nodes,
        "raw_nodes": nodes,
        "edges": edges,
        "raw_edges": edges,
        "missing_edge_endpoints": [],
        "descriptions": {},
        "context": "C6-C7 induced direct-neighbour subgraph from the Observable neuropathic pain graph.",
        "source": OBSERVABLE_NEUROPATHIC_URL,
    }


def save_asia_m(index_rows: list[dict]) -> None:
    graphs = read_graph_definitions()
    asia = normalize_graph(graphs["asia"])
    removed = "either tuberculosis or lung cancer"
    nodes = [n for n in asia["nodes"] if n != removed]
    edges = [(s, d) for s, d in asia["edges"] if s != removed and d != removed]
    graph = {
        "nodes": nodes,
        "raw_nodes": nodes,
        "edges": edges,
        "raw_edges": edges,
        "missing_edge_endpoints": [],
        "descriptions": {k: v for k, v in asia["descriptions"].items() if k != removed},
        "context": "Asia-M derived from Asia by removing the semantically awkward 'either' node.",
        "source": "Derived from Causal_Order_Imperfect_Experts ASIA definition.",
    }
    save_graph_only_dataset("asia_m", graph, index_rows)


def write_root_docs(index_rows: list[dict]) -> None:
    write_csv(
        ROOT / "metadata" / "dataset_index.csv",
        index_rows,
        ["collection", "dataset", "nodes", "edges", "csv_files", "path", "notes"],
    )
    readme = """# Public Causal Discovery Datasets

This folder contains datasets prepared for LLM-guided causal discovery experiments.

Main layout:
- bnlearn/: public Bayesian network repository graphs. Raw graph/CPT files are
  downloaded, and categorical observational CSV files are sampled from the public
  BIF conditional probability tables.
- llm_causal_order_graphs/: graph definitions used by the causal-order LLM
  experiments. These graph sources generally do not publish CPTs or original
  observations, so their CSV files are explicitly synthetic continuous SEM
  samples generated from the graph.
- _sources/: downloaded source pages, repositories, or scripts used to construct
  the prepared datasets.
- dataset_index.csv: one-line inventory for all prepared datasets.

Recommended first experiments:
1. bnlearn/cancer, bnlearn/earthquake, bnlearn/survey, bnlearn/asia
2. bnlearn/child, bnlearn/insurance
3. bnlearn/alarm, bnlearn/mildew
4. llm_causal_order_graphs/neuropathic_paper_repo and
   llm_causal_order_graphs/neuropathic_observable_c6c7

All generated CSV files use a fixed seed for reproducibility.
"""
    (DATA_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    ensure_dir(DATA_DIR)
    ensure_dir(SOURCES_DIR)

    # Keep source landing pages alongside the data for reproducibility.
    for page in ["", "discrete-small.html", "discrete-medium.html"]:
        url = f"{BNLEARN_BASE}/{page}" if page else f"{BNLEARN_BASE}/"
        filename = "index.html" if not page else page
        download(url, SOURCES_DIR / "bnlearn_pages" / filename)

    index_rows = []
    for name, meta in BNLEARN_DATASETS.items():
        save_bnlearn_dataset(name, meta, index_rows)

    graphs = read_graph_definitions()
    graph_names = [
        "cancer",
        "asia",
        "child",
        "earthquake",
        "survey",
        "insurance",
        "covid",
        "alzheimers",
        "neuropathic",
    ]
    for name in graph_names:
        out_name = f"{name}_paper_repo" if name in {"neuropathic"} else f"{name}_paper_repo"
        save_graph_only_dataset(out_name, normalize_graph(graphs[name]), index_rows)

    save_asia_m(index_rows)
    save_graph_only_dataset(
        "neuropathic_observable_c6c7",
        extract_observable_neuropathic(),
        index_rows,
    )

    write_root_docs(index_rows)


if __name__ == "__main__":
    main()
