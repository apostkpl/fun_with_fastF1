from sklearn.metrics import classification_report
import xgboost as xgb
import numpy as np
import pandas as pd
from f1_future_data import pred_cols


# These parameters are estimated using Jupyter and doing some random tests.
# Please do not change here, test first in the ipynb file
HARD_CODED_PARAMS = {'gamma': np.float64(0.05),
                         'learning_rate': np.float64(0.1),
                         'max_depth': 15,
                         'min_child_weight': 58,
                         'n_estimators': 450,
                         'scale_pos_weight': np.float64(16.0)
                         }
def create_model(params = HARD_CODED_PARAMS):
    xgbc = xgb.XGBClassifier(objective='binary:logistic',
                            eval_metric = 'logloss',
                            seed = 42,
                            **params
                            )
    return xgbc

def predict_winner(history_data, history_results, next_race_data, ids, params = HARD_CODED_PARAMS):
    xgbc = create_model(params)
    xgbc.fit(history_data, history_results)
    results_proba = xgbc.predict_proba(next_race_data)
    results_df = ids.copy()
    results_df['Winning Probabilities'] = results_proba[:, 1] # Column 0 probability of losing, 1 probability of winning
    ranked_results = results_df.sort_values(by = 'Winning Probabilities', ascending = False).reset_index(drop = True)
    percent_proba = (ranked_results['Winning Probabilities'] * 100).round(2).astype(str)
    ranked_results['Probability to win'] = percent_proba + "%"
    return ranked_results[['Driver', 'Probability to win']]

def class_report(X, y, ids, params = HARD_CODED_PARAMS):
    X_report = pd.concat([X, y.rename('Winner'), ids], axis = 1)
    X_cols = pred_cols()
    X_train, X_test = X_report[X_report['Year'] != 2025][X_cols], X_report[X_report['Year'] == 2025][X_cols]
    y_train, y_test = X_report[X_report['Year'] != 2025]['Winner'], X_report[X_report['Year'] == 2025]['Winner']
    xgbc = create_model(params)
    xgbc.fit(X_train, y_train)
    report = classification_report(y_test, xgbc.predict(X_test))
    return report

def get_eval_sets(X, y, ids):
    X_report = pd.concat([X, y.rename('Winner'), ids], axis = 1)
    X_cols = pred_cols()
    X_train, X_test = X_report[X_report['Year'] != 2025][X_cols], X_report[X_report['Year'] == 2025][X_cols]
    y_train, y_test = X_report[X_report['Year'] != 2025]['Winner'], X_report[X_report['Year'] == 2025]['Winner']
    return ((X_train, y_train), (X_test, y_test))