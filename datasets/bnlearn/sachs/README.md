# SACHS

Collection: bnlearn Bayesian Network Repository

Source: https://www.bnlearn.com/bnrepository/sachs/

Context: Protein-signaling Bayesian network included as a useful supplementary biological benchmark.

Reference: K. Sachs et al., Causal Protein-Signaling Networks Derived from Multiparameter Single-Cell Data, Science, 2005.

Files:
- raw/: original BIF/DSC/NET/RDA/RDS files downloaded from bnlearn.
- nodes.csv: variables and states parsed from BIF.
- edges.csv: directed ground-truth graph.
- cpts.json: conditional probability tables parsed from BIF.
- samples/: categorical observational CSV samples generated from the BIF CPTs.

Sample sizes: 250, 500, 1000, 5000, 10000

Generation note: the CSV files are synthetic observational samples from the public
Bayesian network parameters, not separately published original observations.
