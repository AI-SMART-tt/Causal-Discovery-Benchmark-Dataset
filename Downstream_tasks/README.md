# Experimental Design: Graph-Informed Charging-Behaviour Models

This document describes the design, mathematical formulation, model settings, and evaluation procedures for a real-data downstream comparison. It does not present experimental results or interpretations.

## 1. Objective and comparison logic

The experiment asks whether the direct parents of a charging-behaviour outcome in an FT-LCD graph provide a useful input specification for a downstream model.

The graph is used to select input variables; the downstream predictive function is then fitted from observations. The experiment does not insert the entire graph into a forecasting architecture or treat graph edges as known causal effects.

Three comparisons are distinguished:

1. **Input-specification comparison:** keep the task and learner fixed, and change only the rule used to select inputs.
2. **Learner comparison:** retain the task-specific input specifications and examine different functional forms.
3. **Outcome comparison:** apply the same comparison framework to categorical charging time and continuous charging-initiation SOC.

The first comparison addresses the downstream value of graph-informed input selection. The other two examine dependence on the learner and outcome. A change associated with switching learners is not, by itself, a benefit attributable to the graph.

## 2. Tasks, observations, and notation

For each event $i$, let $\mathbf{x}_i$ contain the eligible covariates, $v_i$ identify the vehicle, and $d_i$ identify the observation date. Two outcomes are considered:

| Task | Outcome | Statistical treatment |
|---|---|---|
| A | Charging-initiation state of charge (SOC), $y_i^{(A)}\in[0,100]$ | Continuous regression |
| B | Charging-time category, $y_i^{(B)}\in\{0,1,2\}$ | Nominal three-class classification |

For Task B, the categories are off-peak, intermediate, and peak periods. Their numerical labels do not imply equal distances or an ordinal response model. The existing category definitions are retained.

For Task A, the outcome is the SOC actually observed when charging starts. It is not an optimal charging threshold.

For an input-selection rule $s$, let $S_s$ denote the selected variables. A learner $m$, with hyperparameters $\lambda$, is fitted as

$$
\widehat f_{m,s,\lambda}
=
\operatorname{Fit}_m
\left(
\{(\mathbf{x}_{i,S_s},y_i):i\in\mathcal D_{\mathrm{train}}\};
\lambda
\right).
$$

For classification, the output is a probability vector over three categories. For regression, the output is a scalar SOC prediction.

## 3. Sample and chronological partition

The fixed analysis cohort contains 98,578 charging events from 10,000 vehicles. It was reconstructed from the cleaned study observations using the original sampling rule and a vehicle-sampling seed of 42. Exact vehicle-level identity with the original manuscript sample has not been independently established.

The same event cohort and date partitions are used for both tasks:

| Stage | Dates in September 2023 | Events | Role |
|---|---|---:|---|
| Training | 15–17 | 44,099 | Estimate preprocessing, data-dependent input sets, and candidate models |
| Validation | 18–19 | 31,721 | Select hyperparameters |
| Combined fitting | 15–19 | 75,820 | Refit the selected pipelines |
| Later-date evaluation | 29–30 | 22,758 | Evaluate frozen models |

The evaluation sample contains 8,605 vehicles. Combined fitting is the union of training and validation observations, not an additional sample.

The design separates events by date rather than randomly dividing events from the same dates. Vehicles and stations may nevertheless appear in multiple partitions. It therefore evaluates later-date performance within the study population, not strict unseen-vehicle, unseen-station, or external-city generalization.

Cohort preparation checked SOC validity on the seven-day source observations. This eligibility check should not be described as complete non-exposure to evaluation-period SOC.

## 4. Eligible covariates and information restrictions

### 4.1 Task-specific candidate pools

The Task B candidate pool comprises six variables:

- Charging-initiation SOC.
- Station minimum charging price.
- Preassigned driver-shift schedule.
- Rated full-charge driving range.
- Battery capacity.
- Accumulated vehicle mileage.

The Task A candidate pool comprises eight variables:

- Rated full-charge driving range.
- Preassigned driver-shift schedule.
- Charging-start hour.
- Air temperature.
- Fast-charging-pile count at the selected station.
- Battery capacity.
- Accumulated vehicle mileage.
- Precipitation intensity.

Rated range means full-charge range, not remaining range calculated from the target SOC. The shift schedule is a preassigned record. The price covariate is a station minimum-price attribute, not a verified event-specific time-of-use tariff.

### 4.2 Exclusions

The following restrictions are applied before model comparison:

- Exclude driving-behaviour summaries aggregated over all seven observation days, including evaluation dates.
- Exclude driving-profile variables with unverified upstream measurement windows.
- Exclude post-event quantities such as charging duration and end-of-charge SOC.
- Exclude charging-start hour from Task B because the time-category outcome is defined from event time.
- Do not introduce charging-time category into the Task A candidate pool.
- Do not use vehicle or station identifiers as predictors.

These restrictions remove known information-overlap pathways. They do not establish that every retained covariate is available before a future operational decision. Observed event hour, weather, and selected-station attributes are conditional event information.

## 5. Five input specifications

| Specification | Construction | Comparison purpose |
|---|---|---|
| All eligible variables | Use the complete task-specific candidate pool | Reference without input screening |
| Domain-prespecified variables | Use an investigator-defined subset fixed before fitting the extension | Compare with domain-based model specification |
| SHAP top-$k$ | Select the $k$ highest-ranked variables using fitting-data TreeSHAP | Compare with predictive screening at a matched variable count |
| No-LLM graph parents | Select outcome parents from the corresponding no-LLM graph | Compare with the graph-learning implementation without LLM-derived channels |
| FT-LCD graph parents | Select outcome parents from the FT-LCD graph | Evaluate graph-informed input specification |

The domain-prespecified set contains SOC, minimum price, shift schedule, and battery capacity for Task B; it contains rated range, shift schedule, event hour, and battery capacity for Task A. These sets are investigator-defined benchmarks, not independent expert-elicitation results.

### 5.1 Graph-derived input sets

Let $G$ be a learned directed acyclic graph (DAG) and $Y$ its target node. Define

$$
\operatorname{Pa}_{G}(Y)
=
\{X_j:(X_j\rightarrow Y)\in E(G)\}.
$$

The two graph-based specifications are

$$
S_{\mathrm{FT}}=\operatorname{Pa}_{G_{\mathrm{FT}}}(Y),
\qquad
S_{\mathrm{NoLLM}}=\operatorname{Pa}_{G_{\mathrm{NoLLM}}}(Y).
$$

Both graph-learning arms use the same fitting observations, eligible variables, underlying search implementation, and common background prohibitions. The no-LLM arm disables LLM-derived nomination, ancestor/order preferences, and direct prior-score contributions. Its background constraints remain in place; it is not assumption-free discovery.

Existing task-specific semantic priors are retained, while data-dependent graph estimation and weighting use the relevant fitting observations. These historical priors are not new independent validation evidence.

The no-LLM parent count is not forced to match the FT-LCD count. If no inputs are selected, the downstream fallback is an intercept-only model: fitting-sample class frequencies for classification or the fitting-sample outcome mean for regression.

### 5.2 TreeSHAP ranking and matched variable count

SHAP denotes SHapley Additive exPlanations. A fixed LightGBM selector is fitted on the eligible inputs. TreeSHAP explanations use at most 2,048 sampled fitting observations.

For regression, variable importance is

$$
I_j^{(A)}
=
\frac{1}{M}\sum_{i=1}^{M}|\phi_{ij}|,
$$

where $\phi_{ij}$ is the contribution of variable $j$ to the raw regression output.

For classification, importance is averaged across observations and class-specific raw outputs:

$$
I_j^{(B)}
=
\frac{1}{3M}
\sum_{i=1}^{M}\sum_{c=0}^{2}|\phi_{ijc}|.
$$

The contributions satisfy the corresponding additive decomposition,

$$
f_{\mathrm{raw},c}(\mathbf{x}_i)
=
\phi_{0c}+\sum_j\phi_{ijc},
$$

with the class index omitted for regression. Classification contributions are on the raw-score scale, not the probability scale.

At each fitting stage, set

$$
k=|S_{\mathrm{FT}}|,
\qquad
S_{\mathrm{SHAP}}
=
\operatorname{TopK}_{j}(I_j;k).
$$

Importance ties are resolved by variable name. Matching $k$ controls the number of original input variables, not the number of polynomial coefficients or the effective complexity of different learners.

The selector settings are 100 boosting iterations, learning rate 0.05, 15 leaves, maximum depth four, and minimum leaf size 100. The same task- and stage-specific ranking is used across downstream learners.

## 6. Preprocessing

For each continuous covariate $j$, estimate the median $a_j$ from the relevant fitting observations. Missing or non-finite values are replaced as

$$
\widetilde x_{ij}
=
\begin{cases}
x_{ij}, & x_{ij}\text{ is finite},\\
a_j, & \text{otherwise}.
\end{cases}
$$

Compute the mean $\mu_j$ and population standard deviation $\sigma_j$ after imputation, using those same fitting observations, and transform

$$
z_{ij}
=
\frac{\widetilde x_{ij}-\mu_j}{s_j},
\qquad
s_j=
\begin{cases}
\sigma_j,&\sigma_j>0,\\
1,&\sigma_j=0.
\end{cases}
$$

Validation and evaluation observations use the fitted imputation and scaling parameters without re-estimation.

The two shift-schedule codes are represented by a binary indicator rather than standardized as a continuous variable. Unexpected schedule codes are rejected. Quadratic-logit terms receive an additional fitting-data standardization after polynomial expansion.

## 7. Learners and hyperparameter settings

### 7.1 Parametric models

For Task B, multinomial logit models category probabilities as

$$
p_{ic}
=
\frac{
\exp(\beta_{0c}+\boldsymbol{\beta}_c^\top\boldsymbol{\psi}(\mathbf{z}_{i,S_s}))
}{
\sum_{r=0}^{2}
\exp(\beta_{0r}+\boldsymbol{\beta}_r^\top\boldsymbol{\psi}(\mathbf{z}_{i,S_s}))
}.
$$

Ordinary multinomial logit uses the original transformed inputs. Quadratic logit uses

$$
\boldsymbol{\psi}(\mathbf{z})
=
\left(
z_1,\ldots,z_p,\;
z_1^2,\ldots,z_p^2,\;
\{z_jz_l:j<l\}
\right),
$$

followed by the additional standardization described above. Both use L2 regularization.

For Task A, ridge regression estimates

$$
(\widehat\beta_0,\widehat{\boldsymbol{\beta}})
=
\arg\min_{\beta_0,\boldsymbol{\beta}}
\left[
\sum_{i\in\mathcal D_{\mathrm{train}}}
(y_i-\beta_0-\boldsymbol{\beta}^{\top}\mathbf{z}_{i,S_s})^2
+
\alpha\|\boldsymbol{\beta}\|_2^2
\right].
$$

The intercept is not penalized.

### 7.2 Tree ensembles

Random forest regression averages tree predictions; random forest classification averages tree-specific class probabilities:

$$
\widehat y_i
=
\frac{1}{T}\sum_{t=1}^{T}h_t(\mathbf{z}_{i,S_s}),
\qquad
p_{ic}
=
\frac{1}{T}\sum_{t=1}^{T}p_{tc}(\mathbf{z}_{i,S_s}).
$$

LightGBM uses additive boosted trees,

$$
F(\mathbf{z})
=
F_0+\eta\sum_{t=1}^{T}h_t(\mathbf{z}).
$$

For classification, class-specific scores are converted to probabilities by softmax. Regression uses a squared-error training objective; its hyperparameters are nevertheless selected using validation MAE.

### 7.3 Fixed search grids

| Learner | Tasks | Fixed settings | Six candidates per input specification |
|---|---|---|---|
| Quadratic multinomial logit | B | Degree two; L2 penalty; maximum 5,000 iterations; tolerance $10^{-6}$ | $C\in\{10^{-4},10^{-3},10^{-2},10^{-1},1,10\}$ |
| Ridge regression | A | Linear regression with an unpenalized intercept | $\alpha\in\{0.001,0.01,0.1,1,10,100\}$ |
| Random forest | A and B | 200 trees; all inputs available at each split | Maximum depth $\in\{6,12\}$ crossed with minimum leaf size $\in\{50,150,300\}$ |
| LightGBM | A and B | Learning rate 0.05; minimum leaf size 100 | Iterations $\in\{100,250\}$ crossed with leaves $\in\{7,15,31\}$ |

Here, $C$ controls inverse regularization strength, while $\alpha$ is the ridge penalty parameter.

The earlier ordinary multinomial-logit comparison is retained separately. It used the same six-value $C$ grid, maximum 3,000 iterations, and tolerance $10^{-6}$.

The extension contains three learners per task, five specifications per learner, and six candidates per specification: 180 candidate configurations and 30 final models. New learner and primary bootstrap procedures use a fixed seed of 20261020; previously frozen selection procedures retain their earlier settings.

## 8. Training, selection, and refitting workflow

1. **Fix the cohort, dates, candidate pools, learner list, and search grids.**
2. **Construct training-stage inputs.** Fit preprocessing and data-dependent specifications using September 15–17 only.
3. **Fit all candidates.** Evaluate six hyperparameter candidates for every task–learner–specification combination.
4. **Select using validation loss.** Use September 18–19, minimizing Log-loss for Task B or MAE for Task A.
5. **Refit using September 15–19.** Retain the selected hyperparameters and apply the predefined full-pipeline refitting rules.
6. **Freeze all fitted components.**
7. **Evaluate on September 29–30 without further tuning, early stopping, or calibration.**
8. **Calculate paired loss contrasts and vehicle-cluster bootstrap intervals.**

For each learner and specification,

$$
\widehat\lambda_{m,s}
=
\arg\min_{\lambda\in\Lambda_m}
L_{\mathrm{validation}}
\left(\widehat f_{m,s,\lambda}\right).
$$

Each $\Lambda_m$ contains six candidates. Validation losses within $10^{-12}$ are treated as tied, with the earliest grid entry selected. Equal search budgets mean equal candidate counts, not identical runtime or model complexity.

For Task B, the initial-training and combined-fitting input sets, preprocessing, and SHAP rankings are reused from the preceding comparison. They are not reselected for each new learner.

For Task A, preprocessing, both graphs, and SHAP rankings are estimated separately on the initial training data and on the combined fitting data. The matched $k$ is recalculated at each stage. Parent sets may consequently change during the predefined refit; selected hyperparameters are not retuned on evaluation observations.

## 9. Evaluation metrics

Let $N$ be the number of evaluation events. Unless stated otherwise, metrics average over events.

### 9.1 Task B: nominal classification

Let $q_{ic}=\mathbb{1}(y_i=c)$, and let $\widehat c_i=\arg\max_c p_{ic}$.

**Log-loss: primary metric**

$$
\operatorname{LogLoss}
=
-\frac{1}{N}\sum_{i=1}^{N}\log p_{i,y_i}.
$$

Numerical clipping is applied when taking logarithms. Lower values are preferred.

**Multiclass Brier score**

$$
\operatorname{Brier}
=
\frac{1}{N}\sum_{i=1}^{N}
\sum_{c=0}^{2}(p_{ic}-q_{ic})^2.
$$

This is the sum-over-classes convention, without division by three.

**Accuracy**

$$
\operatorname{Accuracy}
=
\frac{1}{N}\sum_{i=1}^{N}
\mathbb{1}(\widehat c_i=y_i).
$$

**Macro-F1**

$$
F1_c
=
\frac{2TP_c}{2TP_c+FP_c+FN_c},
\qquad
\operatorname{MacroF1}
=
\frac{1}{3}\sum_{c=0}^{2}F1_c.
$$

A zero-denominator class receives an F1 of zero. MAE is not calculated from category codes because those codes have no equal-distance interpretation.

### 9.2 Task A: continuous SOC

For every regression model, raw predictions are clipped using the same predefined rule:

$$
\widehat y_i
=
\min\{100,\max\{0,\widehat y_i^{\mathrm{raw}}\}\}.
$$

**Mean absolute error: primary metric**

$$
\operatorname{MAE}
=
\frac{1}{N}\sum_{i=1}^{N}|y_i-\widehat y_i|.
$$

**Root mean squared error**

$$
\operatorname{RMSE}
=
\sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i-\widehat y_i)^2}.
$$

MAE and RMSE are measured in SOC percentage points, not relative percentage error.

**Coefficient of determination**

$$
R^2
=
1-
\frac{\sum_{i=1}^{N}(y_i-\widehat y_i)^2}
{\sum_{i=1}^{N}(y_i-\overline y_{\mathrm{eval}})^2},
\qquad
\overline y_{\mathrm{eval}}
=
\frac{1}{N}\sum_{i=1}^{N}y_i.
$$

Negative values are retained.

**Raw-prediction out-of-range rate**

$$
r_{\mathrm{out}}
=
\frac{1}{N}\sum_{i=1}^{N}
\mathbb{1}
\left(
\widehat y_i^{\mathrm{raw}}<0
\;\text{or}\;
\widehat y_i^{\mathrm{raw}}>100
\right).
$$

The out-of-range rate is calculated before clipping; the other regression metrics use clipped predictions.

### 9.3 Event weighting and vehicle weighting

Let $\mathcal I_v$ contain the evaluation events of vehicle $v$, let $n_v=|\mathcal I_v|$, and let $V$ be the number of evaluation vehicles. Define event-level primary loss as

$$
\ell_i
=
\begin{cases}
-\log p_{i,y_i},&\text{Task B},\\
|y_i-\widehat y_i|,&\text{Task A}.
\end{cases}
$$

The primary event-weighted loss is

$$
L_{\mathrm{event}}
=
\frac{\sum_{v=1}^{V}\sum_{i\in\mathcal I_v}\ell_i}
{\sum_{v=1}^{V}n_v}.
$$

The supplementary vehicle-equal-weighted loss is

$$
L_{\mathrm{vehicle}}
=
\frac{1}{V}\sum_{v=1}^{V}
\left(
\frac{1}{n_v}\sum_{i\in\mathcal I_v}\ell_i
\right).
$$

The first weights each event equally; the second gives each vehicle equal weight. They are two summaries of the same evaluation observations, not independent validations.

### 9.4 Reference models and daily summaries

The classification reference predicts the category frequencies estimated from the combined fitting observations:

$$
p_c^{\mathrm{ref}}
=
\frac{1}{n_{\mathrm{fit}}}
\sum_{i\in\mathcal D_{\mathrm{fit}}}\mathbb{1}(y_i=c).
$$

The regression references predict either the combined-fitting SOC mean or median for every evaluation event. Reference values are not estimated from evaluation dates. Metrics are also calculated separately for September 29 and 30.

## 10. Paired comparisons and uncertainty

### 10.1 Primary loss contrasts

Within each task–learner combination, compare FT-LCD with each of the other four input specifications:

$$
\widehat\Delta_s
=
L_{\mathrm{FT}}-L_s
=
\frac{1}{N}\sum_{i=1}^{N}
\left(\ell_i^{\mathrm{FT}}-\ell_i^{s}\right).
$$

Negative contrasts favour FT-LCD; positive contrasts favour the comparator. All $2\times3\times4=24$ primary contrasts are retained.

### 10.2 Paired vehicle-cluster bootstrap

Repeated events from one vehicle may be dependent. Each bootstrap repetition therefore samples $V$ vehicles with replacement and retains all events belonging to each sampled vehicle. Identical sampled vehicles are used for all compared specifications.

For comparator $s$, define

$$
D_{v,s}
=
\sum_{i\in\mathcal I_v}
(\ell_i^{\mathrm{FT}}-\ell_i^{s}).
$$

Let $w_v^{(b)}$ be the number of times vehicle $v$ is selected in repetition $b$, with

$$
(w_1^{(b)},\ldots,w_V^{(b)})
\sim
\operatorname{Multinomial}
\left(V;\frac1V,\ldots,\frac1V\right).
$$

The event-weighted bootstrap contrast is

$$
\widehat\Delta_s^{*(b)}
=
\frac{\sum_{v=1}^{V}w_v^{(b)}D_{v,s}}
{\sum_{v=1}^{V}w_v^{(b)}n_v}.
$$

This ratio preserves event weighting; it is not the unweighted mean of vehicle-specific contrasts. The procedure uses 1,999 repetitions.

### 10.3 Percentile intervals and multiplicity

Let $Q_p$ denote the empirical $p$-quantile of the bootstrap contrasts. A pointwise 95% interval is

$$
[Q_{0.025},Q_{0.975}].
$$

Within each task–learner combination there are four FT-LCD contrasts. Supplementary Bonferroni intervals use

$$
\left[
Q_{\alpha/(2J)},
Q_{1-\alpha/(2J)}
\right],
\qquad
\alpha=0.05,\quad J=4,
$$

giving quantiles 0.00625 and 0.99375, or 98.75% individual intervals. This is a within-family adjustment, not a global adjustment across all 24 contrasts.

These intervals condition on fixed fitted models and the two evaluation dates. They do not incorporate graph-relearning uncertainty, model-refitting uncertainty, shared date or station effects, or the selection of additional analyses after earlier evaluation results were examined.

### 10.4 Separate post-hoc learner comparison

An additional fixed-prediction diagnostic compares quadratic logit with the earlier ordinary multinomial logit within each of the five input specifications:

$$
\widehat\Delta_s^{\mathrm{quad-MNL}}
=
L_{\mathrm{quad},s}-L_{\mathrm{MNL},s}.
$$

It uses the same paired vehicle-cluster approach and 1,999 repetitions, without refitting or selecting models. Because this is a separate five-comparison family, its Bonferroni intervals use quantiles 0.005 and 0.995, or 99% individual intervals. It is a post-hoc diagnostic, not part of the original four-contrast families or a prespecified confirmatory test.

## 11. Interpretation boundaries

The additional learners and SOC task were introduced after the earlier September 29–30 charging-time evaluation had been examined. SOC from these dates had also previously been read as an input. This extension is therefore exploratory follow-up analysis, not a new untouched test.

Within the extension, evaluation observations are not used for preprocessing estimation, graph learning, SHAP ranking, hyperparameter selection, or prediction calibration. Freezing this workflow does not remove the broader post-holdout analysis-selection limitation.

The experiment concerns target-parent input sets for conditional models of recorded charging events. It does not directly establish complete-DAG accuracy, causal directions, intervention effects, optimal charging thresholds, prospective demand forecasts, or operational policy gains.

No equivalence or non-inferiority margin was specified. An interval containing zero does not establish equivalent performance; an interval excluding zero does not establish practical importance or causal validity. No particular model or input specification is assumed to outperform the alternatives.
