# MILDEW

Collection: bnlearn Bayesian Network Repository

Source: https://www.bnlearn.com/bnrepository/mildew/

Context: Large-parameter agricultural network used by Ban et al.; useful for testing robustness to weaker LLM priors.

Reference: K. G. Olesen et al., A MUNIN network for diagnosis and treatment of mildew in winter wheat, 1989.

Files:
- raw/: original BIF/DSC/NET/RDA/RDS files downloaded from bnlearn.
- nodes.csv: variables and states parsed from BIF.
- edges.csv: directed ground-truth graph.
- cpts.json: conditional probability tables parsed from BIF.
- samples/: categorical observational CSV samples generated from the BIF CPTs.

Sample sizes: 250, 500, 1000, 5000, 10000

Generation note: the CSV files are synthetic observational samples from the public
Bayesian network parameters, not separately published original observations.
