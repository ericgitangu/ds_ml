# Data Science & Machine Learning Interview Prep

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)
[![DS](https://img.shields.io/badge/Data%20Science-Interview%20Prep-blue.svg)](#)
[![ML](https://img.shields.io/badge/Machine%20Learning-Study%20Guide-orange.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg?logo=python&logoColor=white)](#)
[![SQL](https://img.shields.io/badge/SQL-MS%20SQL%20%7C%20PostgreSQL-CC2927.svg?logo=microsoftsqlserver&logoColor=white)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Pipelines-F7931E.svg?logo=scikitlearn&logoColor=white)](#)

A comprehensive, self-contained interview preparation toolkit for **Data Science**, **Machine Learning Engineer**, and **AI/ML** roles. Covers theory, coding challenges, SQL, system design, and interactive practice assessments.

---

## Contents

```
Data_Science_Machine_Learning/
|
|-- ml_ds_study_guide.md          # Concise reference guide (theory + code snippets)
|-- ml_engineer_prep_full.html    # Interactive assessment with 59 questions & timer
|-- README.md
```

---

## Study Guide (`ml_ds_study_guide.md`)

A distilled, interview-ready reference covering eight core domains:

| # | Topic | What's Covered |
|---|-------|----------------|
| 1 | **ML Algorithms** | Regression, classification, unsupervised models; comparison tables with pros/cons/when-to-use; decision framework flowchart |
| 2 | **Model Evaluation** | Classification metrics (accuracy, precision, recall, F1, AUC-ROC, log loss), regression metrics (RMSE, MAE, R-squared), cross-validation strategies, bias-variance tradeoff |
| 3 | **Feature Engineering** | Categorical encoding (label, one-hot, ordinal, target), missing data imputation, scaling/normalization, feature selection (filter, wrapper, embedded) |
| 4 | **Python / Pandas / NumPy** | GroupBy + transform, merge patterns, vectorized ops vs apply, reshaping (pivot/melt), rolling windows, broadcasting, linear algebra |
| 5 | **SQL for Data Science** | Window functions (ROW_NUMBER, RANK, LAG/LEAD, NTILE), CTEs, recursive queries, running totals, deduplication |
| 6 | **Deep Learning** | CNN / RNN / LSTM / Transformer architectures, activation functions, optimizers (SGD, Adam, AdamW), regularization (dropout, batch norm, layer norm) |
| 7 | **Statistics** | Hypothesis testing, p-values, Type I/II errors, confidence intervals, A/B testing methodology, key distributions, Bayes' theorem |
| 8 | **ML System Design** | End-to-end architecture, feature stores, model serving (batch vs online), training pipelines, monitoring & drift detection, worked design questions (recommendations, fraud, search ranking) |

**Bonus sections:**
- Complete **sklearn Pipeline** pattern with `ColumnTransformer` (prevents data leakage)
- **Common Gotchas** checklist for assessments

---

## Interactive Assessment (`ml_engineer_prep_full.html`)

A self-contained HTML app with **59 practice questions**, a built-in timer, and instant feedback. Open in any browser -- no server required.

### Question Breakdown

| Category | Questions | Types | Difficulty |
|----------|-----------|-------|------------|
| **Python** | 7 | Coding + MCQ | Easy - Hard |
| **Sorting** | 3 | Coding + MCQ | Medium |
| **Graphs** | 3 | Coding + MCQ | Medium |
| **Dynamic Programming** | 3 | Coding | Easy - Hard |
| **Data Structures** | 3 | MCQ | Easy - Medium |
| **SQL** | 6 | Coding + MCQ | Easy - Hard |
| **ML / Statistics** | 10 | MCQ | Easy - Hard |
| **Deep Learning** | 6 | MCQ | Medium - Hard |
| **NLP** | 3 | MCQ | Medium - Hard |
| **Computer Vision** | 2 | MCQ | Medium |
| **Feature Engineering** | 2 | MCQ | Medium |
| **Deployment / MLOps** | 2 | MCQ | Medium |
| **Probability** | 2 | MCQ | Medium |
| **Pandas** | 4 | MCQ | Easy - Hard |
| **NumPy** | 2 | MCQ | Medium |
| **scikit-learn** | 2 | MCQ | Medium - Hard |

### Features

- Timed practice mode (30 / 60 / 90 / 120 min options)
- Instant answer validation with detailed explanations
- Difficulty badges (Easy / Medium / Hard)
- Source tags: `CODERBYTE` (platform-style) and `INTERVIEW` (behavioral/conceptual)
- Dark theme, mobile-responsive, zero dependencies

### How to Use

```bash
# Simply open in your browser
open ml_engineer_prep_full.html

# Or serve locally
python -m http.server 8080
# then visit http://localhost:8080/ml_engineer_prep_full.html
```

---

## Quick Start

```bash
git clone <repo-url>
cd Data_Science_Machine_Learning

# Read the study guide (glow for rich rendering, cat as fallback)
command -v glow &>/dev/null && glow ml_ds_study_guide.md || cat ml_ds_study_guide.md

# Take the practice assessment
open ml_engineer_prep_full.html
```

---

## Who Is This For?

- Engineers preparing for **ML Engineer** or **Data Scientist** interviews
- Career switchers building ML/DS fundamentals
- Anyone taking **Coderbyte**, **HackerRank**, or **LeetCode** data science assessments
- Teams looking for onboarding material for junior ML engineers

---

## Topics At a Glance

```
ML Algorithms          Linear/Logistic Regression, Decision Trees, Random Forest,
                       XGBoost, SVM, KNN, Naive Bayes, K-Means, DBSCAN, PCA

Model Evaluation       Confusion Matrix, Precision/Recall/F1, AUC-ROC, RMSE,
                       Cross-Validation, Bias-Variance Tradeoff

Feature Engineering    One-Hot/Label/Target Encoding, Imputation, Scaling,
                       Feature Selection (RFE, L1, Variance Threshold)

Python Essentials      Pandas (groupby, merge, pivot, rolling), NumPy
                       (broadcasting, reshaping, linalg), Vectorized Operations

SQL                    Window Functions, CTEs, Recursive Queries, Aggregations,
                       JOINs, Running Totals, Deduplication

Deep Learning          CNN, RNN/LSTM, Transformers, Attention, Activation
                       Functions, Optimizers, Regularization

Statistics             Hypothesis Testing, A/B Testing, Confidence Intervals,
                       Distributions, Bayes' Theorem, Power Analysis

System Design          Feature Stores, Model Serving, Training Pipelines,
                       Monitoring/Drift, Recommendation/Fraud/Search Systems
```

---

## Contributing

Contributions are welcome! If you'd like to add questions, fix errors, or extend coverage:

1. Fork the repo
2. Create a feature branch (`git checkout -b add-new-questions`)
3. Commit your changes
4. Open a Pull Request

### Ideas for contributions:
- Additional coding challenges (LeetCode-style)
- PyTorch / TensorFlow code snippets
- Case study walkthroughs
- Resume / behavioral interview tips for ML roles

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
