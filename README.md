# Adult Census Income: Data Preparation, Segmentation and Classification

End-to-end data science notebook on the **UCI Adult (Census Income)** dataset, following the **CRISP-DM** approach: explore the data, clean and transform it with every choice justified, segment the population with K-Means, and predict whether a person earns more than 50K$ a year.

> The notebook commentary is written in French.

## Dataset

- **Source:** [UCI Machine Learning Repository, Adult](https://archive.ics.uci.edu/dataset/2/adult) (Becker & Kohavi, 1996), extracted from the 1994 US census.
- **Size:** 48,842 people (32,561 in `adult.data` + 16,281 in `adult.test`), 14 features.
- **Target:** `income` (`<=50K` / `>50K`), with a 76% / 24% class imbalance.
- Check the dataset page for the license terms.

## Project goals

1. **Explore and visualize** the dataset: structure, distributions, relationships with the target, anomalies.
2. **Clean and transform** the data, justifying every method choice.
3. **Segment** the population into socio-economic profiles (clustering).
4. **Classify** income level with appropriate models and metrics.

## Notebook structure

| Section | Content |
|---|---|
| 1. Business Understanding | Problem, objectives, success criteria |
| 2. Data Understanding | Loading, structure, variable types, descriptive statistics |
| 3. Data Exploration | Hidden missing values, class balance, distributions, categorical variables vs. income, correlations |
| 4. Data Preparation | Cleaning, missing values, regrouping, feature engineering, encoding and scaling, export |
| 5. Segmentation | K-Means, choice of k, segment profiling |
| 6. Classification | Baselines, class imbalance, tree-based models, tuning, final test evaluation |
| 7. Evaluation and interpretation | Summary, limits, next steps |

## Key findings and decisions

**Exploration**
- Missing values are hidden as `?` in `workclass`, `occupation` and `native-country` (about 6%, 6% and 2%).
- The missing rows are not random: only 9.4% of them earn more than 50K, against 24.8% for the rest.
- `capital-gain` and `capital-loss` are more than 90% zeros and extremely skewed. All 244 rows with `capital-gain = 99999` are `>50K`.
- The test file labels carry a trailing period (`<=50K.`), which splits each class in two.

**Preparation**

| Problem | Decision |
|---|---|
| `?` in `workclass` / `occupation` | Dedicated `Unknown` category (no row deletion, no mode imputation) |
| `?` in `native-country` | Mode imputation (90% are `United-States`) |
| Target labels | Trailing period removed |
| Duplicates | 29 exact duplicates removed; profile-level duplicates kept |
| `fnlwgt`, `education` | Dropped (sampling weight with no link to income / redundant with `education-num`) |
| Capital variables | `has_capital_*` indicators + `log1p` amounts; the 99,999 rows are kept |
| `native-country` and rare categories | Regrouped (`US` / `Non-US`, etc.) |
| `relationship` | Dropped after an experiment showed no measurable gain |
| Encoding / scaling | One-hot for nominal variables, `StandardScaler` for numeric ones |

**Segmentation**
- A first K-Means run including the capital variables scored a high silhouette (0.511) but was trivial: it only separated "has a gain", "has a loss" and "neither".
- The final run uses `age`, `education-num` and `hours-per-week`, with **k = 6** chosen for interpretability (the elbow and silhouette were inconclusive).
- Segments: young full-time workers, young part-timers, low education, qualified graduates, long hours, active seniors. The share of `>50K` varies from 3% to 47% across segments, although income was not used to build them.

**Classification** (official train/test split, tuning by 5-fold cross-validation on the training set only)

| Model (cross-validation) | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Dummy (always `<=50K`) | 0.759 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic regression | 0.851 | 0.733 | 0.597 | 0.658 | 0.907 |
| Logistic regression (balanced) | 0.810 | 0.571 | 0.848 | 0.682 | 0.907 |
| Decision tree (balanced) | 0.804 | 0.561 | 0.858 | 0.678 | 0.902 |
| Random forest (balanced) | 0.828 | 0.601 | 0.844 | 0.702 | 0.919 |

Final model: tuned random forest (`max_depth=None`, `min_samples_leaf=3`, balanced class weights, without `relationship`), evaluated **once** on the test set:

| Metric (class `>50K`) | Test set |
|---|---|
| F1 | 0.706 |
| Recall | 0.817 |
| Precision | 0.621 |
| ROC-AUC | 0.919 |
| Accuracy | 0.839 |

Test scores are within 0.005 of the cross-validation scores, so there is no sign of overfitting or leakage. The most important features are marital status, age, education level, capital gains and hours worked.

## Repository contents

```
.
├── Lab.ipynb                   # Main notebook 
├── adult_prepared.csv          # Deliverable: cleaned and prepared data
├── adult/                      # Original UCI files
│   ├── adult.data              # Training file
│   ├── adult.test              # Test file
│   ├── adult.names             # Data dictionary
│   └── Index                   # UCI index of the dataset files
├── .gitignore                  # Excludes the regenerated raw backup (adult_raw.csv)
└── README.md
```

### `adult_prepared.csv`

48,813 rows and 44 columns, with no missing values: 5 standardized numeric columns, 2 binary indicators, 35 one-hot columns, the target `income` (0/1, `>50K` = 1) and `source` (official train/test split).

Scaling and encoding in this file are fitted on **all** rows, so it documents the result of the preparation and is not meant as input for a train/test evaluation. In the classification section, these transformations are fitted on the training set only, inside scikit-learn pipelines.

## Getting started

```bash
git clone https://github.com/<your-username>/<your-repo>.git](https://github.com/JasGLe/Adult-Census-Income-DS-LAB.git
cd Adult-Census-Income-DS-LAB
pip install -r requirements.txt
jupyter notebook Lab.ipynb
```

Keep the original UCI files in the `adult/` folder (the notebook reads `adult/adult.data` and `adult/adult.test`), then use **Kernel → Restart & Run All**. The hyperparameter search takes a few minutes. Random steps use `random_state=42`, so results are reproducible.

Running the notebook also writes a raw backup of the merged data (`adult_raw.csv`) and regenerates `adult_prepared.csv`. The raw backup is listed in `.gitignore` because it is rebuilt at each run.

## Requirements

`requirements.txt`:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
scipy
jupyter
```

## Limitations

- 1994 US data: results do not transfer directly to other periods or countries.
- Precision/recall trade-off: class weighting recovers 82% of high earners, but about 38% of `>50K` predictions are wrong.
- `sex` remains a model feature, and the income gap in the data (about 30% of men vs. 11% of women earn `>50K`) can be reproduced. It is an association mixed with hours, occupation and marital status, not evidence of causation. No per-group fairness analysis was done.
- The segmentation uses only three variables, and the segments overlap (silhouette 0.337).

## Possible improvements

- Tune the decision threshold according to the relative cost of false positives and false negatives.
- Try gradient boosting.
- Evaluate performance by sex and other subgroups.
- Test mixed-type clustering (K-Prototypes) to include categorical variables.

## Author

Jasser Yahyaoui

## Acknowledgments

Dataset: Becker, B. & Kohavi, R. (1996). *Adult*. UCI Machine Learning Repository.
