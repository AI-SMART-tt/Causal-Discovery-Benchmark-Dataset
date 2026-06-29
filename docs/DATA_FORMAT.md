# Data format

## Common graph files

### `edges.csv`

A directed ground-truth edge list with one edge per row.

| Column | Type | Meaning |
|---|---|---|
| `source` | string | Parent/cause node identifier |
| `target` | string | Child/effect node identifier |

An entry `A,B` denotes the directed edge `A -> B`. Self-loops are excluded.

### `nodes.csv`

For BNLearn datasets, columns are:

| Column | Type | Meaning |
|---|---|---|
| `node` | string | Variable identifier; also the sample CSV column name |
| `state_count` | integer | Number of categorical states |
| `states` | string | State labels separated by `|` |
| `description` | string | Optional semantic description |

For LLM graph datasets, columns are:

| Column | Type | Meaning |
|---|---|---|
| `node` | string | Variable identifier; also the sample CSV column name |
| `description` | string | Optional source description |
| `in_raw_node_list` | boolean | Whether the node was explicitly present in the source node list |

### `metadata.json`

Machine-readable provenance and dataset statistics. Depending on collection,
it records the source, context/reference, node and edge counts, generation
method, sample sizes, and SHA-256 checksums of original BNLearn files.

## BNLearn collection

- `raw/`: source `.bif`, `.dsc`, `.net`, `.rda`, `.rds` files and compressed
  downloads where available.
- `cpts.json`: variables, categorical states, parent ordering, and conditional
  probability rows parsed from BIF.
- `samples/<name>_n<N>.csv`: categorical observations with one variable per
  column and one observation per row.

Samples are ancestral draws from the published BIF conditional probability
tables in topological order. Available sizes are 250, 500, 1,000, 5,000, and
10,000. They are reproducibly generated benchmark observations, not separately
published real-world patient or survey records.

## LLM causal-order graph collection

- `edges_raw.csv`: edge list as extracted before endpoint cleanup.
- `edges.csv`: cleaned DAG used as ground truth.
- `samples/<name>_synthetic_sem_n<N>.csv`: floating-point observations with
  one graph variable per column and one observation per row.

These graph sources do not bundle public CPTs or observational tables. Samples
therefore come from a synthetic structural equation model. For each edge, a
fixed coefficient is sampled uniformly in magnitude from `[0.35, 1.15]`, with
a 35% probability of a negative sign. Root variables use standard Gaussian
noise. Non-root variables combine their weighted parents with Gaussian noise,
a light `tanh` nonlinearity, and additional Gaussian noise with standard
deviation 0.35. Values are rounded to six decimals. Sizes are 1,000 and 5,000.

## Identifier and parsing guidance

- Treat node identifiers as exact, case-sensitive strings.
- Some LLM graph identifiers contain spaces; use a CSV parser rather than
  splitting lines manually.
- Infer graph nodes from `nodes.csv`, not only from observed edge endpoints.
- Use `edges.csv` for evaluation. `edges_raw.csv` is retained for provenance.
- Empty descriptions mean the source did not supply a description.
