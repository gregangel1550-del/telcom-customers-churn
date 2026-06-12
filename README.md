### telcom-customers-churn

Customers prediction for a telecom company

-- which customers are likely to cancel their subcription in the next 30 days? 

--- how do we reduce customers churn?
 

 The formulation in ML variavble:
 Churn = 1 -> customer left within 30 days 
 Churn = 0 -> customer stayed 
This is a binary clssification problem. 

In this case I'm using tje AUC-ROC as the primary metric:
    The dataset has 26% churn and 74% non-churn. This is a class imbalance. If I use plain accuracy, a model that always predict 'non-churn' owuld score 74% - useless, but numerically impressive. AUC-ROC measures the model's ability to rank churners above non-churners across all decision thresholds, making it robust to class imbalance. 
The Precision-Recall at a threshold of 0.4 as secondary metric: 
     Because of we want to catch as many real churners as possible (recall), but we can't call every single customer, by setting the threshold at 0.4 rather than the default 0.4 because in this business context, a missed churner is more expensive than a false alarm. 

Target: 
    AUC-ROC > 0.82 on the held-out test set




# Problem Statement — Customer Churn Prediction

**Version:** 1.0  
**Status:** Draft — pending stakeholder sign-off before Phase 2

---

## 1. Business Question

A telecom company is losing customers at a rate of ~5% per month. The retention team needs to know, in advance, which customers are likely to cancel so they can intervene with a targeted offer before the customer leaves.

**Core question:** Given a customer's current usage, contract, and billing data, what is the probability that they will cancel their subscription within the next 30 days?

---

## 2. ML Problem Type

This is a **supervised binary classification** problem.

- `Churn = 1` → customer cancelled within 30 days
- `Churn = 0` → customer stayed

---

## 3. Evaluation Metric

**Primary metric:** AUC-ROC  
**Target:** AUC-ROC ≥ 0.82 on the held-out test set

**Why AUC-ROC and not accuracy?**  
The dataset has a class imbalance (~26% churn, ~74% no churn). A model that always predicts "no churn" would score 74% accuracy while being completely useless. AUC-ROC measures the model's ability to rank churners above non-churners across all decision thresholds, making it robust to imbalance.

**Secondary metric:** Precision and Recall at a decision threshold of 0.4  
The threshold is set below 0.5 because missing a real churner (false negative) is more costly to the business than flagging a non-churner for a retention offer (false positive).

---

## 4. Business Value

| Item | Value |
|---|---|
| Monthly customer base | ~7,043 |
| Estimated monthly churn | ~352 customers (~5%) |
| Avg. monthly revenue per customer | ~$65 |
| Monthly revenue lost to churn | ~$22,880 |
| Cost to retain vs. acquire | 5× cheaper to retain |
| Projected monthly saving (model at 40% recovery) | ~$4,576 |

A model that flags the highest-risk customers each week, enabling the retention team to make targeted offers, is estimated to recover ~$4,576/month in otherwise lost revenue.

---

## 5. Constraints & Assumptions

**Capacity:** The retention team can act on a maximum of 500 flagged customers per month. The model output must be a ranked list by churn probability, not a flat binary prediction.

**Explainability:** Predictions must be explainable to non-technical managers. SHAP values will be generated for every high-risk customer flagged.

**Retraining:** Customer behaviour shifts over time due to promotions, price changes, and competitor actions. The model must be retrained monthly on fresh data.

**Data scope:** The model uses only data available at prediction time — no future data leakage. Features are derived from billing, contract, and service usage records.

---

## 6. Ethical Considerations

The dataset contains the features `gender` and `SeniorCitizen`. Before deployment, model performance (precision, recall) will be evaluated separately across these groups to check for discriminatory behaviour. If significant disparity is found, these features will be excluded from training.

---

## 7. Definition of Done

The project is considered successful when:

1. A trained model achieves AUC-ROC ≥ 0.82 on the held-out test set.
2. SHAP explanations are generated and interpretable for the top 10 most important features.
3. A REST API endpoint accepts customer feature data and returns a churn probability score.
4. A monthly retraining schedule is documented.
5. A model card is written summarising performance, limitations, and fairness checks.

---

## 8. Dataset

**Source:** IBM Telco Customer Churn (publicly available on Kaggle)  
**Size:** 7,043 rows × 21 columns  
**Target column:** `Churn` (Yes/No → encoded as 1/0)
