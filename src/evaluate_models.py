# src/evaluate_models.py

import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix

def get_model_outputs(model, test_generator):
    test_generator.reset()
    y_probs = model.predict(test_generator).ravel()
    y_preds = (y_probs > 0.5).astype(int)
    y_true = test_generator.classes
    return y_true, y_probs, y_preds

def ensemble_predictions(base_probs, transfer_probs):
    avg_probs = (base_probs + transfer_probs) / 2
    avg_preds = (avg_probs > 0.5).astype(int)
    return avg_probs, avg_preds

def compute_metrics(y_true, y_probs, y_preds):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    auc_score = auc(fpr, tpr)
    cm = confusion_matrix(y_true, y_preds)
    return fpr, tpr, auc_score, cm
    