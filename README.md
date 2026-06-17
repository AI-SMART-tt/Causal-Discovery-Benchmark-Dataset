# Causal Discovery Benchmark Dataset Structure

## Repository Overview

This repository provides a unified benchmark suite for causal discovery research, containing both BNLearn and LLM-Graphs benchmark families.

```text
causal-discovery-benchmark/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
│
├── datasets/
│   ├── bnlearn/
│   │   ├── asia/
│   │   ├── cancer/
│   │   ├── earthquake/
│   │   ├── survey/
│   │   ├── sachs/
│   │   ├── child/
│   │   ├── insurance/
│   │   ├── alarm/
│   │   └── mildew/
│   │
│   └── llm-graphs/
│       ├── graph01/
│       ├── graph02/
│       ├── graph03/
│       ├── graph04/
│       ├── graph05/
│       ├── graph06/
│       ├── graph07/
│       ├── graph08/
│       ├── graph09/
│       ├── graph10/
│       └── graph11/
│
├── generators/
│   ├── generate_bnlearn.py
│   ├── generate_llm_graphs.py
│   ├── sem_models.py
│   └── utils.py
│
├── configs/
│   ├── sample_sizes.yaml
│   ├── benchmark.yaml
│   └── seeds.yaml
│
├── benchmarks/
│   ├── run_all.sh
│   ├── evaluation.py
│   ├── metrics.py
│   └── experiment_configs/
│
├── docs/
│   ├── dataset_card.md
│   ├── benchmark_protocol.md
│   └── dataset_description.pdf
│
├── metadata/
│   ├── graph_catalog.csv
│   └── benchmark_summary.csv
│
└── results/
    └── example_runs/
```

---

# Dataset Directory Structure

Each graph is stored in an independent directory.

Example:

```text
datasets/
└── bnlearn/
    └── asia/
        ├── dag.graphml
        ├── adjacency.csv
        ├── metadata.json
        ├── n250_seed1.csv
        ├── n250_seed2.csv
        ├── n250_seed3.csv
        ├── n250_seed4.csv
        ├── n250_seed5.csv
        ├── n500_seed1.csv
        ├── n500_seed2.csv
        ├── ...
        └── n10000_seed5.csv
```

---

# Graph Files

## dag.graphml

Ground-truth directed acyclic graph (DAG).

Used by:

* NetworkX
* Graphviz
* Gephi
* Cytoscape

---

## adjacency.csv

Binary adjacency matrix representation.

Example:

```csv
,X1,X2,X3
X1,0,1,0
X2,0,0,1
X3,0,0,0
```

Interpretation:

* X1 → X2
* X2 → X3

---

## metadata.json

Graph metadata.

Example:

```json
{
  "graph_name": "asia",
  "family": "BNLearn",
  "nodes": 8,
  "edges": 8,
  "generator": "CPT",
  "source": "BNLearn"
}
```

---

# Observational Data Files

Each CSV file contains observational samples generated from the corresponding DAG.

Naming convention:

```text
n{sample_size}_seed{seed}.csv
```

Examples:

```text
n250_seed1.csv
n250_seed2.csv
n500_seed1.csv
n1000_seed3.csv
n5000_seed5.csv
```

For BNLearn datasets:

```text
n ∈ {250, 500, 1000, 5000, 10000}
```

For LLM-Graphs datasets:

```text
n ∈ {250, 500, 1000, 5000}
```

Random seeds:

```text
seed ∈ {1,2,3,4,5}
```

---

# Benchmark Families

## BNLearn

The BNLearn family contains 9 published Bayesian-network structures:

| Graph      |
| ---------- |
| Asia       |
| Cancer     |
| Earthquake |
| Survey     |
| Sachs      |
| Child      |
| Insurance  |
| Alarm      |
| Mildew     |

Data generation:

* Conditional Probability Tables (CPTs)
* Ancestral sampling

---

