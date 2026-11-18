# Customer Churn Prediction (Telco)

A supervised machine learning project that predicts whether a telecom customer is likely to churn, built with scikit-learn on the IBM Telco Customer Churn dataset. It covers data cleaning, EDA, leakage-safe preprocessing, a comparison of three algorithms, model interpretation, and a reproducible training and prediction workflow.

## Business problem

Acquiring a new customer usually costs more than keeping an existing one, so telecom companies want to identify customers likely to leave and target retention efforts. This project builds a classifier that flags likely churners and examines the trade-off between missing churners (false negatives) and wasting offers on customers who would have stayed (false positives).

## Dataset

- **Source:** IBM Telco Customer Churn sample dataset (published by IBM; also mirrored on Kaggle). Place `Telco-Customer-Churn.csv` in `data/`.
- **Size:** 7,043 customers, 21 columns. Target: `Churn` (Yes/No).
- **Class balance:** 5,174 non-churners (73.46%) and 1,869 churners (26.54%).
- The data file is not committed to the repository.

## Data cleaning

- `TotalCharges` was stored as text. It had 11 blank values, all belonging to customers with `tenure` = 0 (not yet billed). These were set to 0. The cleaning code raises an error if any blank row has tenure above 0.
- `customerID` was dropped (identifier only).
- `Churn` was encoded as 0/1.
- No true missing values and no fully duplicated rows. After dropping the ID, 22 rows have identical features; they were kept because different customers can share the same plan.
- "No internet service" and "No phone service" were kept as categories.

## Exploratory findings

These are associations in this dataset, not evidence of causes.

| Group | Customers | Churn rate |
|---|---|---|
| Month-to-month contract | 3,875 | 42.71% |
| One-year contract | 1,473 | 11.27% |
| Two-year contract | 1,695 | 2.83% |
| Electronic check | 2,365 | 45.29% |
| Mailed check | 1,612 | 19.11% |
| Bank transfer (automatic) | 1,544 | 16.71% |
| Credit card (automatic) | 1,522 | 15.24% |
| Tenure 0-12 months | 2,186 | 47.44% |
| Tenure 13-24 months | 1,024 | 28.71% |
| Tenure 25-48 months | 1,594 | 20.39% |
| Tenure 49-72 months | 2,239 | 9.51% |

- Churners had a higher mean monthly charge (74.44) than non-churners (61.27), with heavy overlap between the distributions.
- Electronic check users are also much more likely to be on month-to-month contracts (78.2%, versus 35.7% to 38.1% for the two automatic payment methods), so payment method and contract type are confounded.

## Method

- **Split:** 80/20 stratified (5,634 train / 1,409 test), `random_state=42`. Train and test churn rates are both 26.54%.
- **Preprocessing:** a scikit-learn `ColumnTransformer` inside each `Pipeline`. Numeric columns: median imputation and standard scaling. Categorical columns: most-frequent imputation and `OneHotEncoder(handle_unknown="ignore")`. This gives 45 features after encoding. Everything is fitted on training data only, which prevents leakage.
- **Models:** Logistic Regression, Decision Tree (`max_depth=5`) and Random Forest (200 trees, `min_samples_leaf=5`), each with and without `class_weight="balanced"`. No hyperparameter tuning was done.

## Results (test set, 1,409 customers)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8421 |
| Logistic Regression (balanced) | 0.7381 | 0.5043 | 0.7834 | 0.6136 | 0.8416 |
| Decision Tree | 0.7984 | 0.6347 | 0.5668 | 0.5989 | 0.8297 |
| Decision Tree (balanced) | 0.7544 | 0.5260 | 0.7567 | 0.6206 | 0.8318 |
| Random Forest | 0.8041 | 0.6725 | 0.5107 | 0.5805 | 0.8411 |
| Random Forest (balanced) | 0.7615 | 0.5351 | 0.7754 | 0.6332 | 0.8409 |

Predicting "No churn" for everyone would score about 73.5% accuracy, so accuracy alone is misleading. Differences between models (F1 within about 0.03, ROC-AUC within about 0.01) come from a single test split and may be noise.

**Precision** is the share of flagged customers who actually churn. **Recall** is the share of actual churners the model flags.

### Chosen model: Logistic Regression (balanced)

Saved as the final pipeline. Test metrics: accuracy 0.7381, precision 0.5043, recall 0.7834, F1 0.6136, ROC-AUC 0.8416. Confusion matrix: 747 TN, 288 FP, 81 FN, 293 TP. This is a judgment call based on its small train-test accuracy gap (0.0147 versus 0.0699 for the balanced Random Forest) and interpretability, not on being best on every metric.

### Cost sensitivity (illustrative assumptions)

The dataset has no cost information. The numbers below use an **assumed** cost of 500 per missed churner and 50, 150 or 300 per false alarm.

| Model | FP cost 50 | FP cost 150 | FP cost 300 |
|---|---|---|---|
| Random Forest (balanced) | 54,600 | 79,800 | 117,600 |
| Logistic Regression (balanced) | 54,900 | 83,700 | 126,900 |
| Decision Tree (balanced) | 58,250 | 83,750 | 122,000 |
| Logistic Regression | 87,950 | 98,850 | 115,200 |
| Decision Tree | 87,100 | 99,300 | 117,600 |
| Random Forest | 96,150 | 105,450 | 119,400 |

Balanced models are cheapest when a miss costs much more than a false alarm; at the 300 setting the order reverses. The best choice depends on real business costs, which must be supplied.

## Interpretation

- **Logistic Regression coefficients** (numeric features standardized): tenure (-1.154), two-year contract (-0.776), fiber optic (+0.708), month-to-month (+0.658) and DSL (-0.622) had the largest magnitudes. `MonthlyCharges` has a negative coefficient (-0.676) despite churners paying more on average, because it is correlated with tenure, total charges and service type; those coefficients should not be read individually. Several "No internet service" columns are redundant and share one effect.
- **Permutation importance** (drop in test ROC-AUC): tenure and contract type ranked in the top three across the Logistic Regression and Random Forest rankings. Gender, phone service and partner status were at or near zero.
- **Impurity-based importance** (Random Forest) ranked `MonthlyCharges` 6th but its permutation importance was near zero with high variance. Impurity importance is computed on training data and tends to favor continuous features.
- All importances describe what the model relies on, not causes.

## Limitations

- Single train/test split, no cross-validation, no hyperparameter tuning.
- Balanced class weights make predicted probabilities uncalibrated; they rank customers but are not literal churn probabilities.
- Correlated features (tenure, monthly and total charges) make individual coefficients and importances hard to interpret.
- 22 duplicate-feature rows could appear in both train and test (about 0.3% of data).
- Cost figures are placeholders. Results come from one public sample dataset and may not generalize.
- A saved `.pkl` should be loaded with the same scikit-learn version (1.9.1). Only load pickle files you trust.

## Setup

Tested with Python 3.11.3 on Windows.

```powershell
git clone <YOUR-REPO-URL>
cd customer-churn-prediction
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download `Telco-Customer-Churn.csv` into `data/`.

## Usage

```powershell
python -m src.train                          # trains and saves models/churn_model.pkl
python -m src.predict                        # predicts on a built-in example
python -m src.predict path\to\customers.csv  # predicts on a CSV with the original columns
```

Explore the analysis in `notebooks/customer_churn_analysis.ipynb`.

## Project structure

```
customer-churn-prediction/
  data/
  notebooks/customer_churn_analysis.ipynb
  models/            (churn_model.pkl is generated locally; model_metadata.json is committed)
  src/preprocessing.py
  src/train.py
  src/evaluate.py
  src/predict.py
  requirements.txt
  README.md
  .gitignore
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
