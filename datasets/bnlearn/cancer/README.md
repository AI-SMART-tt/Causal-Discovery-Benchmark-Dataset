# CANCER

Collection: bnlearn Bayesian Network Repository

Source: https://www.bnlearn.com/bnrepository/cancer/

Context: Small cancer/outcome Bayesian network; heavily used for LLM pairwise and causal-order examples.

Reference: K. B. Korb and A. E. Nicholson, Bayesian Artificial Intelligence, 2nd edition, Section 2.2.2, 2010.

Files:
- raw/: original BIF/DSC/NET/RDA/RDS files downloaded from bnlearn.
- nodes.csv: variables and states parsed from BIF.
- edges.csv: directed ground-truth graph.
- cpts.json: conditional probability tables parsed from BIF.
- samples/: categorical observational CSV samples generated from the BIF CPTs.

Sample sizes: 250, 500, 1000, 5000, 10000

Generation note: the CSV files are synthetic observational samples from the public
Bayesian network parameters, not separately published original observations.
