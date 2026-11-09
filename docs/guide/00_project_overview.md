# Project overview

## What we built

This project will examine customer churn and build a clear, end-to-end example of a machine-learning project. **Churn** means that a customer stops using or paying for a company's service.

The planned work moves through these stages:

1. Explore customer information with SQL.
2. Check data quality, such as missing or inconsistent values.
3. Prepare the data for analysis and modeling.
4. Build a simple baseline model, then compare it with stronger models.
5. Explore neural networks using TensorFlow and PyTorch.
6. Validate and tune models to check and improve their performance.
7. Use SHAP to explain which information influenced a model's predictions.
8. Share the results in a plain-language report.

This overview describes the project plan; it does not claim that these models or results already exist.

## Why it matters (and which part of a real ML analyst or ML engineer job it matches)

When customers leave, a company can lose future income and may need to spend time and money finding replacements. Understanding which customers might leave can help a company investigate the reasons and decide where to focus its efforts.

This project practices work that can be part of a machine-learning analyst or engineer's job: understanding a business question, checking and preparing data, evaluating models, explaining results, and communicating them clearly. At a company such as HCLTech, an ML analyst could work with customer or operational data, answer business questions with analysis, help assess predictive models, and explain findings to colleagues or clients. Actual responsibilities vary by team and role.

## How it works, step by step

1. Ask the business question: which customers leave, and what patterns appear in the available data?
2. Use SQL to inspect and summarize the data.
3. Check that the data is complete and consistent enough to analyze.
4. Prepare useful inputs for models while keeping track of the changes made.
5. Compare a simple baseline with stronger machine-learning models.
6. Try neural networks with TensorFlow and PyTorch as learning exercises.
7. Validate models on data not used to fit them, then tune settings and compare results.
8. Use SHAP explanations to examine how features relate to individual predictions.
9. Write a plain-language report describing the findings, limitations, and next steps.

## Code walkthrough (explain the important lines)

There is no churn-analysis or model code in this overview. The folders are prepared for later steps: SQL work belongs in `sql/`, Python project code in `src/`, checks in `tests/`, and helper scripts in `tools/`. The project guide builder is documented separately in the code walkthrough for that step.

## Data flow (what goes in, what happens, what comes out)

| Stage | What goes in | What happens | What comes out |
| --- | --- | --- | --- |
| Analysis | Customer data | SQL summaries and data-quality checks | Clearer understanding of the data |
| Preparation | Checked customer data | Clean and organize model inputs | Data ready for modeling |
| Modeling | Prepared data | Train and compare models | Predictions and measured results |
| Explanation and reporting | Model results | Explain predictions and summarize limits | A report people can understand |

The exact data and results will depend on what is available and will only be reported after the relevant work has been run.

## Terms explained

- **Customer churn:** when a customer stops using a company's service.
- **SQL:** a language commonly used to ask questions of data stored in databases.
- **Data quality:** whether data is sufficiently complete, consistent, and correct for its intended use.
- **Baseline model:** a simple starting model used as a comparison for more advanced models.
- **Neural network:** a kind of machine-learning model made of connected calculation units; TensorFlow and PyTorch are tools for building such models.
- **Validation:** checking a model on data set aside from the data used to train it.
- **Tuning:** trying different model settings to see whether performance improves.
- **SHAP:** a method for estimating how input features contribute to a model's prediction.
- **Feature:** an input value about a customer that a model can use.

## What could go wrong

- If customer records are missing, duplicated, or inconsistent, analysis may be misleading.
- A model may learn patterns that do not continue to hold for future customers.
- A model can look accurate overall but still miss many customers who churn.
- A prediction describes a pattern in data; it does not prove why a customer left.
- An explanation is not proof that changing a feature will change the outcome.
- Customer information must be handled responsibly and according to applicable privacy requirements.

## Interview questions with model answers

**Question: What is customer churn?**  
**Model answer:** Customer churn is when a customer stops using or paying for a company's service. A churn analysis looks for patterns in which customers leave and when.

**Question: Why begin with a baseline model?**  
**Model answer:** A baseline gives us a simple reference point. We can compare more complex models against it and check whether their added complexity brings a real improvement.

**Question: Does a model explanation prove why a customer left?**  
**Model answer:** No. An explanation describes how the model used the available input features for its prediction. It does not prove cause and effect or reveal the customer's actual reason.

## Check yourself (2-3 questions)

1. In your own words, what does customer churn mean?
2. Why should a stronger model be compared with a baseline?
3. What can a model explanation tell us, and what can it not prove?
