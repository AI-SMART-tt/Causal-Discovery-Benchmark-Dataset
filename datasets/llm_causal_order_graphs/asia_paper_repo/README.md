# asia_paper_repo

Collection: llm_causal_order_graphs

Source: Causal_Order_Imperfect_Experts graph definitions

Context: model the possible respiratory problems someone can have who has recently visited Asia and is experiencing shortness of breath

Files:
- nodes.csv: node names and descriptions where available.
- edges.csv: directed ground-truth graph used for experiments.
- edges_raw.csv: original edge list before endpoint cleanup, when available.
- samples/: continuous synthetic SEM CSV files generated from the graph.

Important: this graph source does not provide public CPTs or original
observational rows. The CSV files are generated for downstream algorithm
experiments and should be reported as synthetic SEM samples.

