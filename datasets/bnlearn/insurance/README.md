# INSURANCE

Collection: bnlearn Bayesian Network Repository

Source: https://www.bnlearn.com/bnrepository/insurance/

Context: Car-insurance claim-cost network used by Ban et al.; good medium-density benchmark for LLM priors.

Reference: J. Binder, D. Koller, S. Russell and K. Kanazawa, Adaptive probabilistic networks with hidden variables, 1997.

Files:
- raw/: original BIF/DSC/NET/RDA/RDS files downloaded from bnlearn.
- nodes.csv: variables and states parsed from BIF.
- edges.csv: directed ground-truth graph.
- cpts.json: conditional probability tables parsed from BIF.
- samples/: categorical observational CSV samples generated from the BIF CPTs.

Sample sizes: 250, 500, 1000, 5000, 10000

Generation note: the CSV files are synthetic observational samples from the public
Bayesian network parameters, not separately published original observations.
