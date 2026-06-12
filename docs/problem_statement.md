## Customer churn prediction - problem statement

    Business question 
    Which telecom customers are at risk of cancelling their subscription in the next 30 days, and how can we intervene before they leave?

#### Problem type:
--- Binary classification
#### Target variable:
--- Churn = 1/0
#### Primary metric: 
--- AUC-ROC >= 0.82
#### Decision thershold:
--- 0.4(recall-favoured)

#### Business value
Monthly churn cost ~$22,880 in lost revenue. A model that recovers 40% of at-risk customers is estimated to save ~$4,576/month. Retaining one customer costs 5x less than acquiring a new one. 

#### Constraints & assumptions
Retention team can act on at most 500 flagged customers per month. Predictions must be explainable to non-technical managers. Model must be retrained monthly.

#### Definition of success
Model achievs AUC-ROC >= 082 in held-out test data. Output is a ranked list of customers by churn probability, delivered weekly to the retention team via a REST API. 
