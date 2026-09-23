from time import perf_counter
import torch
from src.core.device import synchronize
from .metrics import classification_metrics, count_parameters


@torch.inference_mode()
def evaluate(model, loader, device, num_classes, criterion=None, *, collect_predictions=False):
    """Evaluate logits with an optional per-sample mean loss.

    Timing covers synchronized forward passes only (including the first pass).
    Optional predictions/targets are JSON-safe lists in loader iteration order.
    """
    device = torch.device(device)
    model.to(device)
    model.eval()
    targets, predictions = [], []
    total_loss, elapsed, count = 0.0, 0.0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        synchronize(device)
        start = perf_counter()
        logits = model(inputs)
        synchronize(device)
        elapsed += perf_counter() - start
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Nonfinite evaluation logits; metrics aborted")
        if criterion is not None:
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite evaluation loss; metrics aborted")
            total_loss += loss.item() * labels.size(0)
        targets.append(labels.cpu())
        predictions.append(logits.argmax(1).cpu())
        count += labels.size(0)
    if count == 0:
        raise ValueError("Cannot evaluate an empty loader")
    result = classification_metrics(torch.cat(targets), torch.cat(predictions), num_classes)
    result.update(num_parameters=count_parameters(model),
                  inference_seconds=elapsed, inference_ms_per_sample=1000 * elapsed / count,
                  num_samples=count)
    if criterion is not None:
        result["loss"] = total_loss / count
    if collect_predictions:
        result.update(predictions=torch.cat(predictions).tolist(), targets=torch.cat(targets).tolist())
    return result


validate = evaluate


def evaluate_bundle(model, bundle, config, device):
    return evaluate(model, bundle["loaders"]["test"], device, bundle["metadata"]["num_classes"],
                    torch.nn.CrossEntropyLoss())
