from time import perf_counter
import torch
from src.core.checkpoint import save_checkpoint
from src.core.config import output_dir
from src.core.logger import write_json
from .evaluator import validate


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum, correct, count = 0.0, 0, 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        count += labels.size(0)
        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
    if count == 0:
        raise ValueError("Cannot train on an empty loader")
    return {"loss": loss_sum / count, "accuracy": correct / count}


def fit(model, bundle, config, device, logger):
    """Train with shared loaders; save the first checkpoint with maximal val accuracy."""
    settings = config["training"]
    epochs = settings["epochs"]
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], weight_decay=settings.get("weight_decay", 0.0))
    criterion = torch.nn.CrossEntropyLoss()
    history, best = [], -1.0
    checkpoint = output_dir(config, "checkpoint") / "best.pt"
    start = perf_counter()
    for epoch in range(1, epochs + 1):
        train = train_one_epoch(model, bundle["loaders"]["train"], optimizer, criterion, device)
        val = validate(model, bundle["loaders"]["val"], device, bundle["metadata"]["num_classes"], criterion)
        history.append({"epoch": epoch, "train_loss": train["loss"], "train_accuracy": train["accuracy"],
                        "val_loss": val["loss"], "val_accuracy": val["accuracy"]})
        logger.info("Epoch %s: %s", epoch, history[-1])
        if val["accuracy"] > best:
            best = val["accuracy"]
            save_checkpoint(checkpoint, model, optimizer, epoch, best, config, bundle["metadata"])
    result = {"history": history, "training_seconds": perf_counter() - start,
              "best_validation_accuracy": best, "checkpoint": str(checkpoint),
              "config": config, "dataset": bundle["metadata"], "device": str(device)}
    write_json(output_dir(config, "results") / "training.json", result)
    return result
