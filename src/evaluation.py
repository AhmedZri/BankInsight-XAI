import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve)


def evaluate(y, probabilities, threshold=.5):
    p = np.asarray(probabilities)
    pred = (p >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    return dict(roc_auc=float(roc_auc_score(y, p)), average_precision=float(average_precision_score(y, p)),
        precision=float(precision_score(y, pred, zero_division=0)), recall=float(recall_score(y, pred, zero_division=0)),
        f1=float(f1_score(y, pred, zero_division=0)), confusion=confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        report=classification_report(y, pred, labels=[0, 1], target_names=['Négative', 'Positive'], zero_division=0, output_dict=True),
        roc={'fpr': fpr.tolist(), 'tpr': tpr.tolist()}, pr={'precision': precision.tolist(), 'recall': recall.tolist()})


def fairness_groups(X, y, p, column, threshold=.5):
    numeric_age = column == 'age' and pd.api.types.is_numeric_dtype(X[column])
    groups = pd.cut(X[column], [0, 30, 50, np.inf]).astype(str) if numeric_age else X[column].fillna('manquant').astype(str)
    records = []
    for name in sorted(groups.unique()):
        mask = groups == name
        yy, pp = np.asarray(y)[mask], np.asarray(p)[mask]
        records.append({'groupe': name, 'effectif': int(mask.sum()), 'positifs_réels': int(yy.sum()),
            'taux_ciblage': float((pp >= threshold).mean()), 'rappel': float(recall_score(yy, pp >= threshold, zero_division=0)) if yy.sum() else None})
    return records
