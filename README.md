# Causal Discovery Benchmark Dataset

This repository provides a reproducible benchmark suite for causal structure
learning. It contains **20 directed acyclic graphs** in two complementary
collections:

- **BNLearn (9 networks):** public Bayesian networks with original graph/CPT
  files and categorical observations sampled from their published CPTs.
- **LLM causal-order graphs (11 networks):** graph definitions collected from
  causal-order research and an Observable neuropathic-pain graph, accompanied
  by clearly labelled synthetic continuous SEM observations.

> The CSV observations in this repository are generated benchmark samples.
> They are not claimed to be the original observational records used to create
> the source networks.

## Dataset catalogue

| Collection | Dataset | Nodes | Edges | Sample sizes |
|---|---:|---:|---:|---|
| BNLearn | asia | 8 | 8 | 250, 500, 1,000, 5,000, 10,000 |
| BNLearn | cancer | 5 | 4 | 250, 500, 1,000, 5,000, 10,000 |
| BNLearn | earthquake | 5 | 4 | 250, 500, 1,000, 5,000, 10,000 |
| BNLearn | survey | 6 | 6 | 250, 500, 1,000, 5,000, 10,000 |
| BNLearn | sachs | 11 | 17 | 250, 500, 1,000, 5,000, 10,000 |
| BNLearn | child | 20 | 25 | 250, 500, 1,000, 5,000, 10,000 |
| BNLearn | insurance | 27 | 52 | 250, 500, 1,000, 5,000, 10,000 |
| BNLearn | alarm | 37 | 46 | 250, 500, 1,000, 5,000, 10,000 |
| BNLearn | mildew | 35 | 46 | 250, 500, 1,000, 5,000, 10,000 |
| LLM graphs | cancer_paper_repo | 5 | 4 | 1,000, 5,000 |
| LLM graphs | asia_paper_repo | 8 | 8 | 1,000, 5,000 |
| LLM graphs | child_paper_repo | 20 | 25 | 1,000, 5,000 |
| LLM graphs | earthquake_paper_repo | 5 | 4 | 1,000, 5,000 |
| LLM graphs | survey_paper_repo | 6 | 6 | 1,000, 5,000 |
| LLM graphs | insurance_paper_repo | 27 | 52 | 1,000, 5,000 |
| LLM graphs | covid_paper_repo | 11 | 20 | 1,000, 5,000 |
| LLM graphs | alzheimers_paper_repo | 11 | 19 | 1,000, 5,000 |
| LLM graphs | neuropathic_paper_repo | 24 | 25 | 1,000, 5,000 |
| LLM graphs | asia_m | 7 | 4 | 1,000, 5,000 |
| LLM graphs | neuropathic_observable_c6c7 | 26 | 26 | 1,000, 5,000 |

## Repository layout

```text
datasets/
  bnlearn/<dataset>/
    raw/          # original BIF/DSC/NET/RDA/RDS downloads
    samples/      # generated categorical observational CSV files
    nodes.csv     # variables, states, and optional descriptions
    edges.csv     # directed ground-truth edge list
    cpts.json     # parsed conditional probability tables
    metadata.json
    README.md
  llm_causal_order_graphs/<dataset>/
    samples/      # generated continuous SEM CSV files
    nodes.csv
    edges.csv
    edges_raw.csv
    metadata.json
    README.md
metadata/dataset_index.csv
generators/download_public_datasets.py
docs/DATA_FORMAT.md
docs/SOURCES.md
```

## Quick start

```python
import pandas as pd

data = pd.read_csv("datasets/bnlearn/asia/samples/asia_n1000.csv")
edges = pd.read_csv("datasets/bnlearn/asia/edges.csv")
nodes = pd.read_csv("datasets/bnlearn/asia/nodes.csv")
```

See [DATA_FORMAT.md](docs/DATA_FORMAT.md) for schemas and generation details,
and [SOURCES.md](docs/SOURCES.md) for provenance, references, and reuse notes.
Each dataset directory also contains its own metadata and README.

## Reproducibility

The included generator uses the fixed base seed `20260601`. Running it downloads
the public source files, parses the graphs/CPTs, and regenerates the benchmark
CSVs. It requires Python 3.10+ and only the Python standard library.

```bash
python generators/download_public_datasets.py
```

The script writes to the repository's `datasets/` and `metadata/` directories.
Review those locations before replacing an existing release.

## License and attribution

The repository's MIT license covers original code and documentation. Source
networks and redistributed source files remain subject to their upstream terms
and citations. In particular, the BNLearn repository pages state CC BY-SA and
request attribution to each network's original reference. See
[SOURCES.md](docs/SOURCES.md) before redistribution or publication.
