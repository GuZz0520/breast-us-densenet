from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

def compute_metrics(y_true, y_pred, y_prob, num_classes):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    try:
        macro_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        macro_auc = float("nan")
    return {"acc": acc, "macro_f1": macro_f1, "macro_auc": macro_auc}
