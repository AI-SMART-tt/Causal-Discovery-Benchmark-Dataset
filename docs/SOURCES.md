# Sources and provenance

## BNLearn Bayesian Network Repository

Source repository: <https://www.bnlearn.com/bnrepository/>

The nine networks under `datasets/bnlearn/` were downloaded from their
individual BNLearn repository pages. Original graph/CPT files are retained in
each `raw/` directory; `metadata.json` records their SHA-256 checksums.

| Dataset | Source page | Original reference recorded by this release |
|---|---|---|
| asia | <https://www.bnlearn.com/bnrepository/asia/> | Lauritzen & Spiegelhalter, *Local Computation with Probabilities on Graphical Structures and their Application to Expert Systems* (1988) |
| cancer | <https://www.bnlearn.com/bnrepository/cancer/> | Korb & Nicholson, *Bayesian Artificial Intelligence*, 2nd ed., Sec. 2.2.2 (2010) |
| earthquake | <https://www.bnlearn.com/bnrepository/earthquake/> | Korb & Nicholson, *Bayesian Artificial Intelligence*, 2nd ed., Sec. 2.5.1 (2010) |
| survey | <https://www.bnlearn.com/bnrepository/survey/> | Scutari & Denis, *Bayesian Networks: with Examples in R*, 2nd ed. (2021) |
| sachs | <https://www.bnlearn.com/bnrepository/sachs/> | Sachs et al., *Causal Protein-Signaling Networks Derived from Multiparameter Single-Cell Data* (Science, 2005) |
| child | <https://www.bnlearn.com/bnrepository/child/> | Spiegelhalter & Lauritzen, *Sequential updating of conditional probabilities on directed graphical structures* (1990) |
| insurance | <https://www.bnlearn.com/bnrepository/insurance/> | Binder et al., *Adaptive probabilistic networks with hidden variables* (1997) |
| alarm | <https://www.bnlearn.com/bnrepository/alarm/> | Beinlich et al., *The ALARM Monitoring System: A Case Study with Two Probabilistic Inference Techniques for Belief Networks* (1989) |
| mildew | <https://www.bnlearn.com/bnrepository/mildew/> | Olesen et al., *A MUNIN network for diagnosis and treatment of mildew in winter wheat* (1989) |

BNLearn states that its repository pages are available under CC BY-SA and asks
users to cite the network-specific references. Consult the upstream pages for
the current terms and canonical citation details.

## Causal-order research graph definitions

Primary repository:
<https://github.com/AniketVashishtha/Causal_Order_Imperfect_Experts>

The following graph definitions were extracted from that repository's
`causal_discovery/graphs/definitions.py` source and normalized into node and
edge tables:

- `cancer_paper_repo`
- `asia_paper_repo`
- `child_paper_repo`
- `earthquake_paper_repo`
- `survey_paper_repo`
- `insurance_paper_repo`
- `covid_paper_repo`
- `alzheimers_paper_repo`
- `neuropathic_paper_repo`
- `asia_m` (derived from the ASIA definition by removing the `either` node)

Relevant paper: Vashishtha et al., *Causal Order: The Key to Leveraging
Imperfect Experts in Causal Inference* (2024). Cite the source repository and
paper when using these graph definitions.

## Observable neuropathic-pain graph

`neuropathic_observable_c6c7` is a C6-C7-induced direct-neighbour subgraph
derived from Turuibo's Observable neuropathic-pain graph:

<https://observablehq.com/@turuibo/the-complete-causal-graph-of-neuropathic-pain-diagnosis>

The machine-readable source used during preparation was:
<https://api.observablehq.com/@turuibo/the-complete-causal-graph-of-neuropathic-pain-diagnosis.js?v=3>

## Generated observations

Neither the causal-order graph definitions nor the Observable graph provides
public observations or CPTs. Their CSV observations are explicitly synthetic
SEM data generated for algorithm benchmarking. BNLearn CSV observations are
also generated, but from the public BIF CPTs. See `DATA_FORMAT.md` and the
dataset-level README/metadata files for details.

## Reuse note

The top-level MIT license applies to original repository code and documentation
only. Upstream datasets, graph definitions, and source artifacts retain their
respective attribution and licensing requirements. Users are responsible for
checking the upstream terms appropriate to their use case.
