# Customer Churn Prediction: Methodology

## Objective and data

This project asks whether customer information can help rank and identify customers who are labeled as having churned. It uses the local file `WA_Fn-UseC_-Telco-Customer-Churn.csv` uploaded by BlastChar. The recorded raw-data inspection found 7,043 rows and 21 columns. There were 1,869 churn labels among 7,043 customers, an overall churn rate of 26.54%.

The project notes describe the data as public, but do not identify its publisher or original download location. The filename and local row and column counts are recorded; a more specific origin is not available in the evidence used here.

## Data quality checks and findings

The quality review checked blank strings, unique customer IDs, fully duplicated rows, tenure values, positive monthly charges, valid senior-citizen values, and allowed category labels. In the initial check, 6 checks passed and the blank-string check found 11 whitespace-only `TotalCharges` values. All 11 rows had tenure 0. The value was not a true missing marker: it was a space, so the initial missing-value count did not find it.

The project replaced these 11 values with 0 and converted `TotalCharges` to a numeric column. This is an assumption, not a verified billing fact: zero tenure was taken to mean that no billing cycle had completed. The alternatives recorded were to preserve the values as missing and investigate or fill them later, or to drop the affected rows. Keeping them as missing preserves uncertainty; dropping them loses 11 customers and could introduce bias if those customers differ. The chosen value should be revisited if better billing evidence becomes available. After cleaning, all seven listed quality checks passed.

## Exploratory SQL analysis

The CSV was also loaded into PostgreSQL 17 running in Docker as a raw table whose columns were all stored as text. SQL queries described churn rates by contract type, tenure bucket, and payment method. Because `tenure` was text in the raw table, the tenure-bucket query cast it to an integer before assigning month ranges. These summaries describe patterns in the data; they do not establish causes.

## Preparation

The preparation sequence loads the CSV, cleans `TotalCharges`, converts the Yes/No fields to 0/1, one-hot encodes the remaining categorical fields, and drops `customerID`. One-hot encoding creates indicator columns for categories; `drop_first=True` omits one reference category to avoid making a redundant full set. The prepared table has 30 model features plus the `Churn` target.

The project prepares and encodes the table before the row split. The encoded target is separated from the features, and a stratified 80/20 split uses `random_state=42`. This produced 5,634 training rows and 1,409 test rows, with 30 feature columns in each. Stratification keeps the churn proportion similar in both partitions.

The scaler is not fitted to all rows before cross-validation. In the model comparisons, `StandardScaler` is inside the logistic-regression pipeline; in each validation fold it learns its means and standard deviations from that fold's training rows and applies them to that fold's validation rows. The neural-network comparisons likewise fit a separate scaler on each fold's training rows only. This prevents validation-row values from influencing the learned scaling statistics.

One qualification matters: the project's one-hot encoding is performed before the split, so category levels are discovered from the prepared dataset rather than learned by a fold-specific encoder. The documented workflow treats this as the fixed table-building step, but it is not as strict as fitting all preprocessing inside each training fold. For a production workflow, a fitted encoder with explicit handling of unseen categories inside a pipeline would make this boundary clearer.

## Leakage controls

Three explicit checks were run:

1. **Row overlap:** training and test indices had no intersection. This detects a row assigned to both partitions, but does not detect every form of duplicated or related customer information.
2. **Target exclusion:** `Churn` was absent from both feature tables. This catches direct inclusion of the answer column, but cannot detect every indirect proxy for the target.
3. **Suspicious feature correlation:** absolute training-feature correlations with the training target were checked against 0.9. The highest listed correlations were tenure at 0.346, `InternetService_Fiber optic` at 0.313, and `PaymentMethod_Electronic check` at 0.309, so the check passed. A correlation screen can catch a very strong linear association, but a value below its threshold does not prove that leakage is absent.

All three checks passed. They are targeted safeguards, not proof that every possible source of leakage has been eliminated.

## Model comparison protocol

The baseline and learned models were compared on the training partition with the same five-fold stratified cross-validation, shuffled with `random_state=42`. The reported values are the mean validation scores across folds. Accuracy, precision, recall, and F1 use predicted classes; ROC-AUC measures the ranking of positive cases from probability scores. Precision, recall, and F1 use a zero-division-safe convention.

| Score     | Majority-class baseline | Logistic regression | Random forest | XGBoost | Keras neural network | PyTorch neural network |
| --------- | ----------------------: | ------------------: | ------------: | ------: | -------------------: | ---------------------: |
| Accuracy  |                   0.735 |               0.802 |         0.790 |   0.804 |                0.800 |                  0.803 |
| Precision |                   0.000 |               0.653 |         0.641 |   0.665 |                0.662 |                  0.668 |
| Recall    |                   0.000 |               0.545 |         0.479 |   0.524 |                0.503 |                  0.512 |
| F1        |                   0.000 |               0.594 |         0.548 |   0.586 |                0.570 |                  0.579 |
| ROC-AUC   |                   0.500 |               0.846 |         0.825 |   0.846 |                0.839 |                  0.840 |

These are rounded results from one comparison run, not a statistical test. The baseline predicts the most common class, so its accuracy is not evidence that it identifies churners. Small differences in the table do not establish which model will perform best on future customers.

## Tuning and the decision rule written before the results

The logistic-regression grid tested `C` values 0.01, 0.03, 0.10, 0.30, 1.00, 3.00, and 10.00. The best mean ROC-AUC was 0.8464 at `C=10.00`; `C=1.00` scored 0.8462. The final configuration retained `C=1.00`, because the difference between those two results was only 0.0002 relative to fold-to-fold variation.

The XGBoost grid tested 18 combinations: `max_depth` 2, 3, or 4; `learning_rate` 0.03, 0.05, or 0.10; and `n_estimators` 200 or 300. The written decision rule was to select XGBoost only if its best mean ROC-AUC exceeded the best logistic-regression mean by more than 0.01. The best XGBoost mean was 0.8503, compared with 0.8464 for logistic regression, a reported difference of 0.0040. Since 0.0040 did not exceed 0.01, logistic regression was selected for freezing.

The same folds were used to compare tuning candidates and identify the top result. Selecting the maximum from those results can make the selected cross-validation score somewhat optimistic. The score is a useful comparison, not an independent estimate after model selection.

## Freezing the final model and threshold

The final saved model is a logistic-regression pipeline with a `StandardScaler`, `C=1.0`, and `random_state=42`. It uses 30 features and was fitted on the 5,634 training rows. The threshold was selected as the F1-best cutoff from out-of-fold predictions on training data, before the test set was used. The frozen threshold is 0.30.

Out-of-fold (OOF) means that each training row received a prediction from a fold model that did not train on that row. At the frozen threshold, the saved OOF training estimates were precision 0.5410, recall 0.7672, and F1 0.6346. They are training estimates, not test results.

## Threshold analysis and class weights

The threshold table used out-of-fold training probabilities and evaluated cutoffs from 0.10 through 0.90; F1 peaked at 0.30. Two illustrative cost scenarios produced different preferred cutoffs, 0.10 and 0.35. Their inputs were assumptions rather than company costs, and neither scenario was used to set the frozen threshold.

A separate class-weight comparison used out-of-fold training probabilities. At the 0.5 cutoff, balancing class weights raised recall from 0.545 to 0.801 while ROC-AUC remained 0.846 to three decimals. Across the tested threshold grid, the best-F1 ceiling was similar: 0.635 without balancing and 0.638 with balancing. The operating point changed, but this comparison did not replace the frozen model or its threshold.

## Final evaluation

The frozen model and threshold were evaluated on the held-back test set once. Three safety checks passed: the training row count matched the saved configuration, the test feature names and order matched that configuration, and the test churn rate was within 0.5 percentage points of the training rate (the observed difference was 0.008 percentage points).

At the 0.30 threshold, the test set contained 1,409 customers, including 374 actual churners. The model flagged 543 customers: 282 were actual churners (true positives), 261 were not (false positives), and it missed 92 churners (false negatives). Test accuracy was 0.7495, precision 0.5193, recall 0.7540, F1 0.6150, and ROC-AUC 0.8418. The saved results also report average precision 0.6324.

Derived from those stated test counts, not from a new run: an always-no-churn prediction would be correct for 1,035 of 1,409 customers (1,409 minus 374), giving accuracy 1,035 / 1,409 = 0.7346; the model's test accuracy of 0.7495 is about 1.5 percentage points higher.

Bootstrap intervals were calculated using 1,000 resamples with `random_state=42`. The 95% percentile intervals were 0.8188–0.8624 for ROC-AUC, 0.4770–0.5599 for precision, 0.7119–0.7946 for recall, and 0.5760–0.6497 for F1. These intervals describe sampling variation from resampling this test sample with the fitted model and threshold held fixed. They do not cover uncertainty from choosing a model or threshold, changes in the customer population, future time periods, or all other sources of error.

## Explainability

SHAP (SHapley Additive exPlanations) values were computed for training rows using the saved logistic pipeline, its already-fitted scaler, and a linear explainer. For each row, the base value plus the feature contributions equals the model's raw decision score in log-odds. This additivity check is useful because a mismatch can reveal a problem with the feature order, scaling, or calculation. The largest observed absolute difference was 2.6645352591e-15, and the feature shape and name checks passed.

The SHAP contributions describe how this fitted model combines its inputs; they do not establish why a customer behaves as they do or that a feature causes churn. A review found seven exactly identical no-internet-service indicator columns. Other input columns also overlap. Therefore, individual coefficients or SHAP contributions for correlated inputs should not be read as isolated effects.

## Known limitations and honest notes

- The data is described in the project notes as public, not company data. Its publisher and original download location were not recorded in the evidence used for this document.
- The final threshold was selected to maximize training OOF F1, not accuracy. On this test set, its accuracy of 0.7495 is about 1.5 percentage points above the derived always-no-churn accuracy of 0.7346. This is a modest accuracy gain and does not by itself establish the value of the model; recall and precision describe different parts of its behavior.
- Out-of-fold segment analysis found recall of 0.287 for customers with 49–72 months of tenure. This is a training-only diagnostic and shows that the model misses many long-standing churners under the frozen rule.
- Gender is present as a model input. The recorded raw-data churn rates were 26.9% for female customers and 26.2% for male customers; `gender_Male` did not appear among the 15 largest mean absolute SHAP features. OOF recall was 0.779 for female customers and 0.756 for male customers. This is only an initial group comparison, not proof of fairness or lack of harm.
- Seven “no internet service” indicators are exact duplicates. `PhoneService` and `MultipleLines_No phone service` are perfectly negatively correlated, and tenure and `TotalCharges` have correlation 0.830. Redundancy makes single-feature interpretations less reliable.
- The business cost and customer-value scenarios explored earlier were explicitly illustrative assumptions. They were not used to select the frozen threshold; the business must supply real costs and goals before choosing an operational cutoff.
- The threshold favors a balance measured by F1, not maximum accuracy. The accuracy comparison with the majority-class baseline is modest, and the protocols differ as noted above.
- The top tuning score is subject to selection optimism because the same cross-validation folds were used to compare candidates and choose the top setting.
- Fixed seeds make the workflow more repeatable, but do not guarantee bit-for-bit identical neural-network runs. Keras and PyTorch use different random-number implementations and may differ in initialization, dropout, batch order, and numerical operations.

## Reproducibility

The recorded environment was Python 3.12.6. Relevant installed package versions from the virtual environment were: NumPy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1, XGBoost 3.4.1, TensorFlow 2.21.0, Keras 3.15.1, PyTorch 2.15.0, SHAP 0.54.1, Matplotlib 3.11.2, SQLAlchemy 2.1.8, psycopg2-binary 2.9.13, python-dotenv 1.2.4, and pytest 9.2.0.

The documented run sequence for recreating the project outputs is:

1. Prepare and split data: `.\.venv\Scripts\python.exe -m src.prepare_data`.
2. Clean the raw data and display quality checks on the cleaned data: `.\.venv\Scripts\python.exe -m src.clean_data`.
3. Run leakage checks: `.\.venv\Scripts\python.exe -m src.leakage_checks`. The command `.\.venv\Scripts\python.exe -m src.run_quality_report` loads the raw data and rewrites `reports/data_quality_report.md` with raw-data results; adding `--cleaned` writes the raw-versus-cleaned comparison to `reports/data_quality_report_after_cleaning.md` without overwriting either report.
4. Compare the six models on training data: `.\.venv\Scripts\python.exe -m src.torch_model`.
5. Run the logistic and XGBoost searches on training data: `.\.venv\Scripts\python.exe -m src.tuning`.
6. Freeze the selected model and threshold using training data: `.\.venv\Scripts\python.exe -m src.final_model`.
7. Run the final test evaluation once, after freezing: `.\.venv\Scripts\python.exe -m src.final_evaluation`.
8. Calculate the saved test confidence intervals: `.\.venv\Scripts\python.exe -m src.bootstrap_intervals`.
9. Training-only diagnostics and explanations can then be reproduced with `.\.venv\Scripts\python.exe -m src.error_analysis` and `.\.venv\Scripts\python.exe -m src.explainability`.

The automated test suite is run with `.\.venv\Scripts\python.exe -m pytest -v`; it collected and passed 50 tests in the recorded run.

The data split, cross-validation shuffles, model random states, bootstrap procedure, and neural-network seed use 42 where configured. The split uses an 80/20 stratified division; cross-validation uses five stratified folds. The frozen test evaluation should not be repeated as a tuning step.
