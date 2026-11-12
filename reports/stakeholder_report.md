# Customer churn model: stakeholder report

## Purpose

This project asks whether customer information can help identify customers who may leave, so a team can review those predictions and consider appropriate support. The report separates patterns seen in the customer data from how well the frozen model performed on previously unseen customers.

## Key findings about who churns

Across the full raw dataset, the observed churn rate was **26.5%**. The tables below show churn rates by contract, months of tenure, and payment method. These are descriptive patterns in this dataset, not explanations of what caused a customer to leave.

| Contract | Churn rate |
| --- | ---: |
| Month-to-month | 42.7% |
| One year | 11.3% |
| Two year | 2.8% |

| Tenure (months) | Churn rate |
| --- | ---: |
| 0-12 | 47.4% |
| 13-24 | 28.7% |
| 25-48 | 20.4% |
| 49-72 | 9.5% |

| Payment method | Churn rate |
| --- | ---: |
| Bank transfer (automatic) | 16.7% |
| Credit card (automatic) | 15.2% |
| Electronic check | 45.3% |
| Mailed check | 19.1% |

## How well the model works on customers it has never seen

The final evaluation used **1,409** held-out customers, including **374** who actually churned. At the frozen cutoff of **0.30**, the model flagged **543** customers. It caught **282** actual churners and missed **92**. Of the customers flagged, **261** did not churn.

**Recall** means the share of actual churners the model caught: test recall was **75.4%**, with a 95% interval from **71.2%** to **79.5%**. **Precision** means the share of flagged customers who actually churned: test precision was **51.9%**, with a 95% interval from **47.7%** to **56.0%**. These intervals describe uncertainty from sampling customers for this test evaluation; they do not include variation from training a new model.

## The trade-off

The frozen cutoff is **0.30**: a customer at or above that predicted probability is flagged for review. A lower cutoff can catch more potential churners, but it can also flag more customers who would not churn and may receive an unnecessary contact. A higher cutoff generally means fewer contacts but can miss more churners. The appropriate balance depends on the real cost of contacting customers and the value of retaining them; business owners must provide those inputs before choosing a cutoff for operational use. No dollar savings are estimated here.

## Limits and cautions

- The data is a public dataset, not company customer data. Results may not transfer to a particular company.
- The group rates are patterns, not causes. They do not prove why customers leave.
- The raw-data churn rate was **26.9%** for female customers and **26.2%** for male customers. In the training-only ranking of the 15 largest mean absolute SHAP values, the gender feature does not appear. A next version should drop it and then be evaluated again.
- Some input columns duplicate one another. When inputs overlap, individual feature contributions should not be read as isolated effects.
- On out-of-fold training predictions, recall for customers with 49-72 months of tenure was **28.7%**. This figure comes from training data split into folds, where each row is predicted by a fold model that did not train on that row; it is not a test-set subgroup result.
- Gender fairness was only given a first look by comparing out-of-fold training recall: female **77.9%**, male **75.6%**. Similarity or difference in this one measure is not proof of fairness.
- The confidence intervals cover test-sample variation only. They do not account for retraining the model, changes over time, different company populations, or every source of uncertainty.

| Training group | Share of actual churners caught (recall) |
| --- | ---: |
| 0-12 months | 87.8% |
| 13-24 months | 84.2% |
| 25-48 months | 67.8% |
| 49-72 months | 28.7% |

## Suggested next steps

- Test a next model version without gender, and assess it using an appropriate evaluation plan.
- Review duplicated and overlapping input columns in a future development step; do not change this frozen model based on the test results.
- Investigate why long-standing customers are often missed, using training analysis first and a separate evaluation plan for any proposed changes.
- Ask business owners to define contact costs, customer value, and acceptable contact volume before selecting an operational cutoff.
- Validate the approach on suitable company data and continue monitoring prediction quality and fairness before using it to guide customer outreach.
