# Online Shopper Intent: Prediction, Segmentation and Anomaly Detection

A semester-one course project (MSc in Data Science and Artificial Intelligence,
RKMVERI) that predicts whether an online shopping session ends in a purchase,
groups sessions into customer segments, and flags unusual sessions.

## Dataset
UCI Online Shoppers Purchasing Intention dataset: 12,330 sessions, 18 columns.
After removing 125 duplicates: 12,205 sessions. Target: `Revenue` (about 16% buyers).

## Methods
- **Classification:** Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM, K-NN
  with stratified 5-fold cross-validation, class weights, hyper-parameter tuning,
  and a decision threshold (0.698) chosen from out-of-fold F1.
- **Segmentation:** PCA (13 components for 90% variance) and K-means (K = 4).
- **Anomaly detection:** Isolation Forest (300 trees, contamination 0.05).
- **Interpretation:** gain and permutation importance, `PageValues` ablation.

## Results
| Item | Value |
|---|---|
| Best classifier | XGBoost |
| Test ROC-AUC | 0.938 |
| Test F1 (tuned threshold) | 0.689 |
| Clusters | 4 (silhouette 0.274) |
| Anomalies | 611 (5.0%) |

## How to run
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
jupyter notebook shoppers_intent.ipynb   # run all cells from top to bottom
```
Data file: `online_shoppers_intention.csv` (place it in the same folder).

## Using the saved model
```python
predict_sessions(new_sessions_dataframe)
```
Returns purchase probability, decision and anomaly flag for each session.

## Files
- `shoppers_intent.ipynb`: full analysis code
- `Project_Report_Pallab_Nath_B2630045.pdf`: project report
- `models/`: saved model files (joblib)

## Author
Pallab Nath (B2630045)
