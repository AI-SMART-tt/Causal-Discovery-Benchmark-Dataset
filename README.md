# Causal Discovery Dataset

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

