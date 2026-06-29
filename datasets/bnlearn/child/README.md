# CHILD

Collection: bnlearn Bayesian Network Repository

Source: https://www.bnlearn.com/bnrepository/child/

Context: Congenital heart disease network; medium-sized and important in both Vashishtha et al. and Ban et al.

Reference: D. J. Spiegelhalter and S. L. Lauritzen, Sequential updating of conditional probabilities on directed graphical structures, 1990.

Files:
- raw/: original BIF/DSC/NET/RDA/RDS files downloaded from bnlearn.
- nodes.csv: variables and states parsed from BIF.
- edges.csv: directed ground-truth graph.
- cpts.json: conditional probability tables parsed from BIF.
- samples/: categorical observational CSV samples generated from the BIF CPTs.

Sample sizes: 250, 500, 1000, 5000, 10000

Generation note: the CSV files are synthetic observational samples from the public
Bayesian network parameters, not separately published original observations.
