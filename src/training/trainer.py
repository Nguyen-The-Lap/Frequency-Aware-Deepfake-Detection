from pathlib import Path

import torch
from torch import nn


class Trainer:
    def __init__(self, model, device, learning_rate=1e-4, weight_decay=1e-4, patience=5):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=1)
        self.patience = patience

    def _epoch(self, loader, training):
        self.model.train(training)
        total_loss = correct = total = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                if training:
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item() * labels.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total += labels.size(0)
        if total == 0:
            raise ValueError("Cannot train or validate on an empty loader")
        return {"loss": total_loss / total, "accuracy": correct / total}

    def fit(self, train_loader, val_loader, epochs, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        best, stale, history = float("inf"), 0, []
        for epoch in range(1, epochs + 1):
            train = self._epoch(train_loader, True)
            val = self._epoch(val_loader, False)
            self.scheduler.step(val["loss"])
            history.append({"epoch": epoch, "train": train, "val": val,
                            "learning_rate": self.optimizer.param_groups[0]["lr"]})
            if val["loss"] < best:
                best, stale = val["loss"], 0
                torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(), "val_loss": best}, checkpoint_path)
            else:
                stale += 1
                if stale >= self.patience:
                    break
        return history

