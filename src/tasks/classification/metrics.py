import torch


def classification_metrics(targets, predictions, num_classes):
    matrix = torch.bincount(targets * num_classes + predictions, minlength=num_classes**2).reshape(num_classes, num_classes)
    tp = matrix.diag().float()
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = torch.where(denominator > 0, 2 * tp / denominator.clamp_min(1), 0)
    return {"accuracy": tp.sum().item() / max(1, matrix.sum().item()),
            "macro_f1": f1.mean().item(), "confusion_matrix": matrix.tolist()}
