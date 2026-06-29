# ASIA / Lung Cancer

Collection: bnlearn Bayesian Network Repository

Source: https://www.bnlearn.com/bnrepository/asia/

Context: Respiratory diagnosis network for a patient who recently visited Asia; used by Vashishtha et al. and Ban et al.

Reference: S. Lauritzen and D. Spiegelhalter, Local Computation with Probabilities on Graphical Structures and their Application to Expert Systems, 1988.

Files:
- raw/: original BIF/DSC/NET/RDA/RDS files downloaded from bnlearn.
- nodes.csv: variables and states parsed from BIF.
- edges.csv: directed ground-truth graph.
- cpts.json: conditional probability tables parsed from BIF.
- samples/: categorical observational CSV samples generated from the BIF CPTs.

Sample sizes: 250, 500, 1000, 5000, 10000

Generation note: the CSV files are synthetic observational samples from the public
Bayesian network parameters, not separately published original observations.
