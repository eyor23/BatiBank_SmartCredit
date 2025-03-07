import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, classification_report

def train_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """Trains and evaluates a model."""
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print(f"\n{model_name} Results:")
    print(classification_report(y_test, predictions))
    print("ROC-AUC:", roc_auc_score(y_test, probabilities))
    print("F1-score:", f1_score(y_test, predictions))