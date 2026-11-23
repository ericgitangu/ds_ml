# ML Engineer / Data Science Technical Assessment Study Guide

---

## 1. ML Algorithms

### Supervised - Regression

| Algorithm | When to Use | Pros | Cons |
| ----------- | ------------ | ------ | ------ |
| **Linear Regression** | Linear relationship, interpretability needed | Fast, interpretable, no tuning | Assumes linearity, sensitive to outliers |
| **Ridge / Lasso** | Linear + regularization needed | Handles multicollinearity (Ridge), feature selection (Lasso L1 drives coefficients to 0) | Still assumes linearity |
| **XGBoost / Gradient Boosting** | Tabular data, competitions, max performance | Handles nonlinearity, missing values, feature importance built-in | Slower to train, can overfit, less interpretable |

### Supervised - Classification

| Algorithm | When to Use | Pros | Cons |
| ----------- | ------------ | ------ | ------ |
| **Logistic Regression** | Binary/multi classification, interpretable baseline | Fast, probabilistic output, good baseline | Linear decision boundary |
| **Decision Tree** | Interpretability, non-linear boundaries | No scaling needed, handles mixed types, visual | Overfits easily, unstable (high variance) |
| **Random Forest** | General-purpose, reduce DT variance | Robust, parallel training, handles high-dim | Slow prediction, less interpretable than single tree |
| **XGBoost** | Structured/tabular data, best performance | State-of-art on tabular, built-in regularization | Hyperparameter-sensitive, slower training |
| **SVM** | Small-medium data, high-dim, clear margin | Effective in high dimensions, kernel trick for nonlinearity | Doesn't scale well (>100k samples), no native probability |
| **KNN** | Small data, simple baseline, anomaly detection | No training phase, intuitive | Slow at prediction (O(n)), curse of dimensionality, needs scaling |
| **Naive Bayes** | Text classification, spam filtering | Very fast, works well with small data | Assumes feature independence |

### Unsupervised

| Algorithm | When to Use | Pros | Cons |
| ----------- | ------------ | ------ | ------ |
| **K-Means** | Clustering when k is known, spherical clusters | Fast, simple, scalable | Must specify k, assumes spherical clusters, sensitive to initialization |
| **DBSCAN** | Arbitrary-shape clusters, outlier detection | No need to specify k, finds outliers | Struggles with varying densities, sensitive to eps/min_samples |
| **PCA** | Dimensionality reduction, visualization, noise reduction | Reduces features, removes multicollinearity | Linear only, components hard to interpret |

### Key Decision Framework

```yaml
Tabular data, max performance  -->  XGBoost / LightGBM
Need interpretability          -->  Logistic Regression / Decision Tree
Small dataset                  -->  SVM, KNN, Naive Bayes
Image data                     -->  CNN
Sequence / text data           -->  RNN/LSTM or Transformer
Unsupervised grouping          -->  K-Means, DBSCAN
Reduce dimensions              -->  PCA, t-SNE (viz only)
```

---

## 2. Model Evaluation

### Classification Metrics

| Metric | Formula | When to Use |
| -------- | --------- | ------------ |
| **Accuracy** | (TP+TN) / Total | Balanced classes only |
| **Precision** | TP / (TP+FP) | Cost of false positives is high (spam filter) |
| **Recall (Sensitivity)** | TP / (TP+FN) | Cost of false negatives is high (cancer detection) |
| **F1 Score** | 2 *(P*R) / (P+R) | Imbalanced classes, balance P and R |
| **AUC-ROC** | Area under TPR vs FPR curve | Compare models, threshold-independent evaluation |
| **Log Loss** | -mean(y*log(p) + (1-y)*log(1-p)) | When you care about calibrated probabilities |

**Confusion Matrix Layout:**

```yaml
                 Predicted
              Pos      Neg
Actual Pos  [ TP  |   FN ]
Actual Neg  [ FP  |   TN ]
```

### Regression Metrics

| Metric | Formula | Notes |
| -------- | --------- | ------- |
| **RMSE** | sqrt(mean((y-y_hat)^2)) | Penalizes large errors more, same units as target |
| **MAE** | mean(abs(y-y_hat)) | Robust to outliers, same units as target |
| **R-squared** | 1 - SS_res/SS_tot | Proportion of variance explained, 0-1 (can be negative) |
| **MAPE** | mean(abs((y-y_hat)/y)) * 100 | Percentage error, undefined when y=0 |

### Cross-Validation

```python
# K-Fold (standard)
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='f1')

# Stratified K-Fold - maintains class distribution (use for imbalanced)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Time Series - no shuffling, respects temporal order
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

### Bias-Variance Tradeoff

```yaml
Total Error = Bias^2 + Variance + Irreducible Error

High Bias (Underfitting)          High Variance (Overfitting)
- Train error high                - Train error low
- Train ~ Test (both bad)         - Test error >> Train error
- Model too simple                - Model too complex

Fixes for underfitting:           Fixes for overfitting:
- More features                   - More training data
- More complex model              - Regularization (L1/L2/dropout)
- Less regularization             - Simpler model
- Longer training                 - Early stopping
                                  - Cross-validation
                                  - Feature selection
```

---

## 3. Feature Engineering

### Encoding Categorical Variables

```python
# Label Encoding - ordinal categories (low/med/high)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['size_encoded'] = le.fit_transform(df['size'])  # 0, 1, 2

# One-Hot Encoding - nominal categories (red/blue/green)
pd.get_dummies(df, columns=['color'], drop_first=True)  # drop_first avoids multicollinearity

# Ordinal Encoding - when order matters
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder(categories=[['low', 'medium', 'high']])

# Target Encoding - high cardinality categoricals (zip codes)
# Replace category with mean of target variable (watch for leakage - use CV)
```

### Handling Missing Data

```python
# Detect
df.isnull().sum()
df.isnull().mean()  # fraction missing per column

# Simple imputation
df['col'].fillna(df['col'].median(), inplace=True)  # numeric - median is robust to outliers
df['col'].fillna(df['col'].mode()[0], inplace=True)  # categorical - mode

# Sklearn imputer (use in pipelines to avoid leakage)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')  # 'mean', 'median', 'most_frequent', 'constant'

# Indicator for missingness (can be predictive)
df['col_missing'] = df['col'].isnull().astype(int)

# KNN Imputer - uses similar rows
from sklearn.impute import KNNImputer
```

**Key rule:** Fit imputer on train set only, then transform both train and test.

### Scaling / Normalization

```python
# StandardScaler (Z-score) - mean=0, std=1, use for most algorithms
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# MinMaxScaler - scales to [0,1], use when you need bounded range
from sklearn.preprocessing import MinMaxScaler

# RobustScaler - uses median/IQR, robust to outliers
from sklearn.preprocessing import RobustScaler

# Log transform - right-skewed distributions
import numpy as np
df['log_col'] = np.log1p(df['col'])  # log1p handles zeros
```

**Which algorithms need scaling?** SVM, KNN, logistic regression, neural networks, PCA.
**Which don't?** Tree-based models (decision tree, random forest, XGBoost).

### Feature Selection Methods

```python
# Filter methods - statistical tests, fast
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
selector = SelectKBest(f_classif, k=10)

# Wrapper methods - recursive feature elimination
from sklearn.feature_selection import RFE
rfe = RFE(estimator=model, n_features_to_select=10)

# Embedded methods - built into model
# L1 (Lasso) regularization drives coefficients to zero
# Tree-based feature_importances_
importances = model.feature_importances_

# Variance threshold - remove low-variance features
from sklearn.feature_selection import VarianceThreshold
```

---

## 4. Python / Pandas / NumPy

### Pandas Essentials

```python
# ---------- GroupBy ----------
df.groupby('category')['value'].agg(['mean', 'count', 'sum'])
df.groupby(['cat1', 'cat2']).agg(
    avg_val=('value', 'mean'),
    max_val=('value', 'max')
).reset_index()

# Transform - returns same-shaped output (great for group-level features)
df['group_mean'] = df.groupby('category')['value'].transform('mean')
df['normalized'] = df.groupby('cat')['val'].transform(lambda x: (x - x.mean()) / x.std())

# ---------- Merge / Join ----------
pd.merge(df1, df2, on='key', how='left')          # left, right, inner, outer
pd.merge(df1, df2, left_on='id', right_on='key')  # different column names
df1.merge(df2, on=['key1', 'key2'])                # composite key

# ---------- Apply ----------
df['new'] = df['col'].apply(lambda x: x**2)              # element-wise
df['combined'] = df.apply(lambda row: row['a'] + row['b'], axis=1)  # row-wise

# PREFER vectorized operations over apply (10-100x faster):
df['new'] = df['col'] ** 2
df['combined'] = df['a'] + df['b']
df['category'] = np.where(df['val'] > 0, 'positive', 'negative')  # vectorized conditional

# ---------- Reshaping ----------
# Pivot: long to wide
df.pivot_table(index='date', columns='product', values='sales', aggfunc='sum')

# Melt: wide to long
pd.melt(df, id_vars=['id'], value_vars=['col1', 'col2'], var_name='feature', value_name='value')

# ---------- Window / Rolling ----------
df['rolling_mean'] = df['value'].rolling(window=7).mean()
df['cumsum'] = df.groupby('cat')['value'].cumsum()
df['rank'] = df.groupby('cat')['value'].rank(ascending=False)
df['pct_change'] = df['value'].pct_change()

# ---------- Useful Patterns ----------
df.value_counts('col', normalize=True)        # frequency distribution
df.nlargest(5, 'value')                       # top 5
df.query("age > 30 and city == 'NYC'")        # readable filtering
pd.cut(df['age'], bins=[0,18,35,65,100])      # binning
pd.qcut(df['score'], q=4)                     # quantile binning
```

### NumPy Essentials

```python
# Reshaping
a = np.array([1,2,3,4,5,6])
a.reshape(2, 3)          # 2x3 matrix
a.reshape(-1, 1)         # column vector (infer rows)

# Broadcasting - operations on different shapes
# (3,4) + (4,)   -> OK, broadcasts row
# (3,4) + (3,1)  -> OK, broadcasts column
# (3,4) + (3,)   -> ERROR

# Linear algebra
np.dot(A, B)              # matrix multiply
A @ B                     # same thing
np.linalg.inv(A)          # inverse
np.linalg.eig(A)          # eigenvalues/eigenvectors

# Common operations
np.argmax(a)              # index of max value
np.argsort(a)             # indices that would sort
np.concatenate([a, b])    # join arrays
np.stack([a, b], axis=0)  # stack arrays along new axis
np.unique(a, return_counts=True)
np.clip(a, 0, 1)          # clamp values
```

---

## 5. SQL for Data Science (Quick Reference)

```sql
-- Window Functions
SELECT *,
    ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) as rn,
    RANK() OVER (ORDER BY salary DESC) as rnk,           -- gaps after ties
    DENSE_RANK() OVER (ORDER BY salary DESC) as drnk,    -- no gaps
    LAG(salary, 1) OVER (ORDER BY hire_date) as prev_sal,
    LEAD(salary, 1) OVER (ORDER BY hire_date) as next_sal,
    SUM(salary) OVER (PARTITION BY dept) as dept_total,
    AVG(salary) OVER (ORDER BY hire_date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as moving_avg,
    NTILE(4) OVER (ORDER BY salary) as quartile
FROM employees;

-- CTE (Common Table Expression)
WITH ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY dept ORDER BY salary DESC) as rn
    FROM employees
)
SELECT * FROM ranked WHERE rn = 1;  -- top earner per dept

-- Recursive CTE (hierarchies)
WITH RECURSIVE org AS (
    SELECT id, name, manager_id, 1 as level FROM employees WHERE manager_id IS NULL
    UNION ALL
    SELECT e.id, e.name, e.manager_id, o.level + 1 FROM employees e JOIN org o ON e.manager_id = o.id
)
SELECT * FROM org;

-- Useful patterns
-- Running total
SUM(amount) OVER (ORDER BY date ROWS UNBOUNDED PRECEDING)

-- Percent of total
amount * 1.0 / SUM(amount) OVER ()

-- Year-over-year comparison
LAG(revenue) OVER (ORDER BY year)

-- Deduplication
DELETE FROM t WHERE id NOT IN (SELECT MIN(id) FROM t GROUP BY unique_cols);

-- HAVING vs WHERE: WHERE filters rows, HAVING filters groups
```

---

## 6. Deep Learning Basics

### Architectures

#### CNN (Convolutional Neural Networks) - Image data**

```yaml
Input -> [Conv -> ReLU -> Pool] x N -> Flatten -> FC -> Output

Conv layer:  Learns spatial filters (edges, textures, shapes)
Pooling:     Reduces spatial dimensions (MaxPool most common)
Key params:  kernel_size, stride, padding, num_filters
Output size: (W - K + 2P) / S + 1
```

#### RNN / LSTM - Sequential data (time series, text)**

```yaml
RNN problem:  Vanishing/exploding gradients on long sequences
LSTM solves:  Gates (forget, input, output) control information flow
GRU:          Simplified LSTM (2 gates instead of 3), often similar performance
Bidirectional: Process sequence both forward and backward
```

#### Transformers - Modern NLP, increasingly everything

```yaml
Key innovation:  Self-attention (no sequential processing -> parallelizable)
Attention:       Q, K, V matrices; Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
Multi-head:      Multiple attention heads capture different relationships
Positional enc:  Since no recurrence, position info added explicitly

Encoder-only:    BERT (classification, embeddings)
Decoder-only:    GPT (generation)
Encoder-Decoder: T5, original Transformer (translation)
```

### Activation Functions

| Function | Formula | Use Case | Notes |
| ---------- | --------- | ---------- | ------- |
| **ReLU** | max(0, x) | Hidden layers (default) | Fast, sparse, dying ReLU problem |
| **Leaky ReLU** | max(0.01x, x) | Fix dying ReLU | Small negative slope |
| **Sigmoid** | 1/(1+e^-x) | Binary output layer | Vanishing gradient, output (0,1) |
| **Tanh** | (e^x-e^-x)/(e^x+e^-x) | Hidden layers (RNN) | Output (-1,1), zero-centered |
| **Softmax** | e^xi / sum(e^xj) | Multi-class output | Outputs sum to 1 |
| **GELU** | x * Phi(x) | Transformers | Smooth approximation of ReLU |

### Optimizers

```yaml
SGD:           Simple, needs good LR. Add momentum for faster convergence.
Adam:          Adaptive LR per parameter (default choice). Combines momentum + RMSprop.
AdamW:         Adam + decoupled weight decay (preferred for transformers).
Learning rate: Too high -> diverge. Too low -> slow. Use schedulers (cosine, warmup).
```

### Regularization

```yaml
Dropout:       Randomly zero out neurons during training (p=0.1-0.5). Reduces co-adaptation.
Batch Norm:    Normalize activations per mini-batch. Speeds training, slight regularization.
               Applied BEFORE activation (Conv -> BN -> ReLU).
Layer Norm:    Normalize across features (not batch). Used in transformers.
Weight Decay:  L2 penalty on weights. AdamW decouples this from gradient update.
Early Stopping: Monitor val loss, stop when it stops improving.
Data Aug:      Random crops, flips, rotations (images). Mixup, cutout.
```

### Key Hyperparameters

```yaml
Learning rate:   Most important. Start with 3e-4 (Adam), 1e-2 (SGD with momentum).
Batch size:      Larger = faster but may generalize worse. 32-256 typical.
Architecture:    Depth vs width tradeoff. Deeper = more abstract features.
```

---

## 7. Statistics

### Hypothesis Testing Framework

```yaml
1. State hypotheses:  H0 (null: no effect)  vs  H1 (alternative: there is an effect)
2. Choose significance level:  alpha = 0.05 (typically)
3. Compute test statistic and p-value
4. Decision:  p < alpha -> reject H0    p >= alpha -> fail to reject H0
```

**p-value:** Probability of observing data at least as extreme as what we got, assuming H0 is true. It is NOT the probability that H0 is true.

**Type I Error (False Positive):** Reject H0 when it's true. Rate = alpha.
**Type II Error (False Negative):** Fail to reject H0 when it's false. Rate = beta.
**Power = 1 - beta:** Probability of detecting a real effect. Increase with larger sample size.

### Common Tests

| Scenario | Test |
| ---------- | ------ |
| Compare 2 means (normal, known var) | Z-test |
| Compare 2 means (normal, unknown var) | t-test (independent or paired) |
| Compare >2 means | ANOVA (one-way, two-way) |
| Compare proportions | Chi-squared test |
| Non-normal / ordinal data | Mann-Whitney U, Wilcoxon signed-rank |
| Correlation | Pearson (linear), Spearman (monotonic) |
| Normality check | Shapiro-Wilk, KS test |

### Confidence Intervals

```yaml
CI = point_estimate +/- z * standard_error

95% CI with z=1.96:   x_bar +/- 1.96 * (s / sqrt(n))

Interpretation: If we repeated the experiment many times, 95% of the
constructed intervals would contain the true parameter.
It does NOT mean 95% probability the parameter is in this interval.
```

### A/B Testing

```yaml
1. Define metric (conversion rate, revenue, etc.)
2. Calculate required sample size BEFORE the test:
   - Effect size (minimum detectable effect, e.g., 2% lift)
   - Significance level (alpha = 0.05)
   - Power (1 - beta = 0.80)

3. Randomly assign users to control (A) vs treatment (B)
4. Run test until predetermined sample size is reached (don't peek!)
5. Analyze with appropriate test (z-test for proportions, t-test for means)

Common pitfalls:
- Peeking at results early (inflates false positive rate) -> use sequential testing
- Multiple comparisons -> Bonferroni correction (alpha / num_tests)
- Simpson's paradox -> segment analysis
- Novelty/primacy effects -> run long enough
- Network effects -> use cluster randomization
```

### Key Distributions

```yaml
Normal:      Continuous, symmetric. Central Limit Theorem: sample means are normal.
Bernoulli:   Single binary trial (coin flip). P(X=1) = p.
Binomial:    n independent Bernoulli trials. P(X=k) = C(n,k) * p^k * (1-p)^(n-k).
Poisson:     Count of events in fixed interval. P(X=k) = (lambda^k * e^-lambda) / k!
Exponential: Time between Poisson events. Memoryless property.
Uniform:     Equal probability across range.

Bayes' Theorem: P(A|B) = P(B|A) * P(A) / P(B)
```

---

## 8. System Design for ML

### End-to-End ML System Architecture

```yaml
Data Sources -> Ingestion -> Feature Store -> Training Pipeline -> Model Registry
                                  |                                     |
                                  v                                     v
                            Online Features  <----  Model Serving  ->  Monitoring
                                                         |
                                                    Predictions -> Logging -> Retraining trigger
```

### Feature Store

```yaml
Purpose:      Single source of truth for features. Prevents training/serving skew.
Offline store: Historical features for training (data warehouse, S3/Parquet).
Online store:  Low-latency features for serving (Redis, DynamoDB).
Key features:  Point-in-time correctness, feature versioning, lineage tracking.
Tools:         Feast (open source), Tecton, SageMaker Feature Store.
```

### Model Serving

```yaml
Batch prediction:   Run periodically, store results. Simple, high throughput.
                    Use when: latency tolerance is high, predictions can be precomputed.

Online prediction:  Real-time API. REST/gRPC endpoint.
                    Use when: need low latency, input only known at request time.

Patterns:
- Model-as-service: Dedicated endpoint (SageMaker, Vertex AI, self-hosted).
- Embedded model:   Model bundled in application (edge, mobile).
- Model mesh:       Multiple models behind unified API.

Optimization:
- Quantization (FP32 -> INT8): 2-4x speedup, minimal accuracy loss.
- Distillation: Train small model to mimic large model.
- ONNX Runtime: Framework-agnostic optimized inference.
- Batching: Group requests for GPU efficiency.
- Caching: Cache predictions for repeated inputs.
```

### Training Pipeline

```yaml
Components:
1. Data validation   - Schema checks, distribution drift detection (Great Expectations)
2. Preprocessing     - Feature transforms, consistent with serving
3. Training          - Hyperparameter tuning, experiment tracking (MLflow, W&B)
4. Evaluation        - Metrics on holdout set, comparison to baseline/champion
5. Model registry    - Version control, staging/production promotion (MLflow)
6. CI/CD for ML      - Automated testing, shadow deployment, canary rollout

Orchestration: Airflow, Kubeflow Pipelines, Prefect
```

### Monitoring & Drift

```yaml
Data drift:     Input distribution changes (feature distributions shift).
                Detect: KS test, PSI (Population Stability Index), JS divergence.

Concept drift:  Relationship between features and target changes.
                Detect: Monitor prediction distribution, actual vs predicted metrics.

Model performance decay: Accuracy/precision/recall degrading over time.
                Detect: Compare metrics to baseline on recent labeled data.

What to monitor:
- Prediction distribution (mean, variance, percentiles)
- Feature distributions (per feature, compare to training data)
- Latency and throughput
- Error rates
- Data quality (nulls, out-of-range, schema violations)

Response:
- Alerts when drift exceeds threshold
- Automatic retraining trigger
- Shadow mode for new models before promotion
- A/B testing new model vs champion
```

### Common Interview Design Questions

```yaml
Q: "Design a recommendation system"
   -> Collaborative filtering vs content-based vs hybrid
   -> Candidate generation (ANN) -> Ranking (ML model) -> Re-ranking (business rules)
   -> Cold start: content-based for new items, popular items for new users
   -> Feature store for user/item features, real-time signals

Q: "Design a fraud detection system"
   -> Highly imbalanced -> SMOTE, class weights, or anomaly detection
   -> Real-time scoring needed -> online serving
   -> Features: transaction velocity, location anomaly, device fingerprint
   -> Ensemble: rules + ML model
   -> Monitor false positive rate (customer friction) and false negative rate (fraud loss)

Q: "Design a search ranking system"
   -> Two-stage: retrieval (BM25/embeddings + ANN) then ranking (LTR model)
   -> Features: query-doc relevance, freshness, popularity, user personalization
   -> Evaluation: NDCG, MRR
   -> Online: A/B test with interleaving
```

---

## Quick Reference: Sklearn Pipeline Pattern

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define transforms per column type
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Full pipeline - prevents data leakage
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10, None]
}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1')
grid.fit(X_train, y_train)
print(f"Best score: {grid.best_score_:.3f}")
print(f"Best params: {grid.best_params_}")
```

---

## Common Gotchas in Assessments

1. **Data leakage:** Fitting scaler/imputer on full data before splitting. Always fit on train only.
2. **Imbalanced classes:** Don't use accuracy. Use F1, AUC-ROC, or precision-recall. Apply SMOTE only to training data.
3. **Multicollinearity:** Drops model interpretability. Use VIF, correlation matrix, or L1 regularization.
4. **Feature scaling:** Forgetting to scale for distance-based algorithms (SVM, KNN, PCA).
5. **Target leakage:** Including features derived from the target or future data.
6. **Train/test contamination:** Any preprocessing step that uses info from test set.
7. **Random state:** Set seeds for reproducibility (`random_state=42`).
8. **Pandas copy warning:** Use `.copy()` when slicing DataFrames to avoid `SettingWithCopyWarning`.
