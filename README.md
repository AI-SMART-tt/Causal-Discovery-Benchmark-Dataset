# Causal Discovery Benchmark Dataset

## Overview

This repository provides a unified benchmark suite for causal discovery research.

The benchmark contains 20 directed acyclic graph (DAG) structures from two widely used families:

| Family     | Graphs | Nodes | Edges |
| ---------- | ------ | ----- | ----- |
| BNLearn    | 9      | 5–37  | 4–52  |
| LLM-Graphs | 11     | 5–27  | 4–52  |
| Total      | 20     | 5–37  | 4–52  |

For each graph, observational datasets are generated under multiple sample sizes and random seeds.

### Sample Sizes

BNLearn:

* n = 250
* n = 500
* n = 1000
* n = 5000
* n = 10000

LLM-Graphs:

* n = 250
* n = 500
* n = 1000
* n = 5000

### Random Seeds

Each configuration is generated using:

```text
seed = {1,2,3,4,5}
```

### Benchmark Scale

* 20 DAGs
* 5 random seeds
* 24 experimental configurations
* 10,680 completed runs

---

## Dataset Families

### BNLearn

The BNLearn family contains canonical Bayesian-network benchmarks:

* Asia
* Cancer
* Earthquake
* Survey
* Sachs
* Child
* Insurance
* Alarm
* Mildew

Ground-truth DAGs are obtained from published Bayesian-network structures and observational samples are generated through conditional probability table (CPT) sampling.


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
│   │   │   ├── dag.graphml
│   │   │   ├── adjacency.csv
│   │   │   ├── metadata.json
│   │   │   ├── n250_seed1.csv
│   │   │   ├── n250_seed2.csv
│   │   │   └── ...
│   │   │
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
│       ├── ...
│       └── graph11/
│
├── generators/
│   ├── generate_bnlearn.py
│   ├── generate_llm_graphs.py
│   └── sem_models.py
│
├── benchmarks/
│   ├── configs/
│   ├── run_all.sh
│   └── evaluation.py
│
├── docs/
│   ├── dataset_description.pdf
│   └── benchmark_protocol.md
│
└── results/
    └── example_runs/
