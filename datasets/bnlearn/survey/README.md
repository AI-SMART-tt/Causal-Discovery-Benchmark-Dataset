# SURVEY

Collection: bnlearn Bayesian Network Repository

Source: https://www.bnlearn.com/bnrepository/survey/

Context: Hypothetical transport survey network used in Vashishtha et al.

Reference: M. Scutari and J.-B. Denis, Bayesian Networks: with Examples in R, 2nd edition, 2021.

Files:
- raw/: original BIF/DSC/NET/RDA/RDS files downloaded from bnlearn.
- nodes.csv: variables and states parsed from BIF.
- edges.csv: directed ground-truth graph.
- cpts.json: conditional probability tables parsed from BIF.
- samples/: categorical observational CSV samples generated from the BIF CPTs.

Sample sizes: 250, 500, 1000, 5000, 10000

Generation note: the CSV files are synthetic observational samples from the public
Bayesian network parameters, not separately published original observations.
