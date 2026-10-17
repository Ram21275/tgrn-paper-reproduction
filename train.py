from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ImportError:  # Minimal/offline environments can train without a progress bar.
    def tqdm(iterable, **_: object):  # type: ignore[no-redef]
        return iterable

from tgrn import TgRN, TgRNConfig
from tgrn.data import MultimodalPickleDataset
from tgrn.masking import apply_random_missingness
from tgrn.metrics import regression_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TgRN on precomputed multimodal features")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/run"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model: TgRN, loader: DataLoader, device: torch.device, rates: dict[str, float]) -> dict[str, float]:
    model.eval()
    predictions, targets = [], []
    generator = torch.Generator(device=device).manual_seed(2025)
    for raw in loader:
        batch = move(raw, device)
        full = {m: batch[m] for m in ("text", "audio", "vision")}
        valid = {m: batch[f"{m}_valid"] for m in full}
        corrupted, _ = apply_random_missingness(full, valid, rates, generator=generator)
        output = model(**corrupted, **{f"{m}_valid": valid[m] for m in full})
        predictions.append(output.prediction.cpu())
        targets.append(batch["label"].cpu())
    return regression_metrics(torch.cat(predictions).numpy(), torch.cat(targets).numpy())


def main() -> None:
    args = parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8"))
    set_seed(int(settings.get("seed", 42)))
    device = torch.device(args.device)
    config = TgRNConfig.from_dict(settings["model"])
    train_set = MultimodalPickleDataset(args.data, "train")
    valid_set = MultimodalPickleDataset(args.data, "valid")
    test_set = MultimodalPickleDataset(args.data, "test")
    workers = int(settings["training"].get("num_workers", 0))
    loader_options = {"num_workers": workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_set, batch_size=settings["training"]["batch_size"], shuffle=True, **loader_options)
    valid_loader = DataLoader(valid_set, batch_size=settings["training"]["batch_size"], shuffle=False, **loader_options)
    test_loader = DataLoader(test_set, batch_size=settings["training"]["batch_size"], shuffle=False, **loader_options)
    model = TgRN(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["training"]["learning_rate"], weight_decay=settings["training"].get("weight_decay", 0.0))
    rates = settings["missing_rates"]
    args.output.mkdir(parents=True, exist_ok=True)
    best_mae, stale = float("inf"), 0
    for epoch in range(1, settings["training"]["epochs"] + 1):
        model.train()
        running = 0.0
        generator = torch.Generator(device=device).manual_seed(settings.get("seed", 42) + epoch)
        for raw in tqdm(train_loader, desc=f"epoch {epoch}"):
            batch = move(raw, device)
            full = {m: batch[m] for m in ("text", "audio", "vision")}
            valid = {m: batch[f"{m}_valid"] for m in full}
            corrupted, missing = apply_random_missingness(full, valid, rates, generator=generator)
            output = model(**corrupted, **{f"{m}_valid": valid[m] for m in full})
            loss, _ = model.loss(output, batch["label"], full, missing, valid)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
        metrics = evaluate(model, valid_loader, device, rates)
        print(json.dumps({"epoch": epoch, "train_loss": running / max(len(train_loader), 1), **metrics}))
        if metrics["mae"] < best_mae:
            best_mae, stale = metrics["mae"], 0
            torch.save({"model": model.state_dict(), "config": config.to_dict(), "metrics": metrics}, args.output / "best.pt")
        else:
            stale += 1
            if stale >= settings["training"].get("patience", 8):
                break
    (args.output / "config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test_metrics = evaluate(model, test_loader, device, rates)
    (args.output / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print("test:", json.dumps(test_metrics))


if __name__ == "__main__":
    main()
