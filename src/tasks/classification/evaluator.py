from time import perf_counter
import torch
from src.core.device import synchronize
from .metrics import classification_metrics


@torch.inference_mode()
def evaluate(model, loader, device, num_classes, criterion=None):
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
        if criterion is not None:
            total_loss += criterion(logits, labels).item() * labels.size(0)
        targets.append(labels.cpu())
        predictions.append(logits.argmax(1).cpu())
        count += labels.size(0)
    if count == 0:
        raise ValueError("Cannot evaluate an empty loader")
    result = classification_metrics(torch.cat(targets), torch.cat(predictions), num_classes)
    result.update(num_parameters=sum(p.numel() for p in model.parameters()),
                  inference_seconds=elapsed, inference_ms_per_sample=1000 * elapsed / count,
                  num_samples=count)
    if criterion is not None:
        result["loss"] = total_loss / count
    return result


validate = evaluate


def evaluate_bundle(model, bundle, config, device):
    return evaluate(model, bundle["loaders"]["test"], device, bundle["metadata"]["num_classes"])
