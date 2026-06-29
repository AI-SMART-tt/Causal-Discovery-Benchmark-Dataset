# ALARM

Collection: bnlearn Bayesian Network Repository

Source: https://www.bnlearn.com/bnrepository/alarm/

Context: A Logical Alarm Reduction Mechanism for patient monitoring; used by Ban et al. as a medium-sized medical-style benchmark.

Reference: I. Beinlich et al., The ALARM Monitoring System: A Case Study with Two Probabilistic Inference Techniques for Belief Networks, 1989.

Files:
- raw/: original BIF/DSC/NET/RDA/RDS files downloaded from bnlearn.
- nodes.csv: variables and states parsed from BIF.
- edges.csv: directed ground-truth graph.
- cpts.json: conditional probability tables parsed from BIF.
- samples/: categorical observational CSV samples generated from the BIF CPTs.

Sample sizes: 250, 500, 1000, 5000, 10000

Generation note: the CSV files are synthetic observational samples from the public
Bayesian network parameters, not separately published original observations.
