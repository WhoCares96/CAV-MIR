import json
import os

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from cavmir.training.network import CAVNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_samples_from_batch(batch, device=DEVICE) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = batch["npz"]["embedding"].float().to(device)
    targets = batch["npz"]["target"].float().to(device)
    return inputs, targets


def get_sample_shapes(dataset: DataLoader, device=DEVICE) -> tuple[int, int]:
    batch = next(iter(dataset))
    inputs, targets = get_samples_from_batch(batch, device)
    input_shape = inputs.shape[-1]
    target_shape = targets.shape[-1]
    return input_shape, target_shape


def fit_cav_model(
    model: CAVNetwork,
    train_dataset: DataLoader,
    val_dataset: DataLoader,
    out_files_dir: str,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 5,
    device=DEVICE,
    verbose_steps: int = 1,
) -> None:
    def print_if_verbose(message: str, epoch: int) -> None:
        if (epoch + 1) % verbose_steps == 0 and verbose_steps > 0:
            print(message)

    model = model.to(device)

    os.makedirs(out_files_dir, exist_ok=True)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_loss_history = []
    val_loss_history = []

    # Training Phase
    for epoch in range(num_epochs):
        print_if_verbose(f"Epoch {epoch + 1}/{num_epochs}", epoch)

        model.train()
        epoch_loss = 0.0

        for i, batch in enumerate(train_dataset):
            inputs, targets = get_samples_from_batch(batch, device)

            optimizer.zero_grad()

            outputs = model.forward_train(inputs)

            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / (i + 1)

        print_if_verbose(f"Epoch {epoch + 1} Train Loss: {avg_train_loss:.4f}", epoch)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i, batch in enumerate(val_dataset):
                inputs, targets = get_samples_from_batch(batch, device)

                outputs = model.forward_train(inputs)

                loss = criterion(outputs, targets)

                val_loss += loss.item()

            avg_val_loss = val_loss / (i + 1)

        print_if_verbose(
            f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}", epoch
        )

        scheduler.step(avg_val_loss)

        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            model_file_name = save_model(model, out_files_dir)
            if verbose_steps > 0:
                print_if_verbose(f"Model saved to {model_file_name}", epoch)

        else:
            epochs_no_improve += 1

        train_loss_history.append(avg_train_loss)
        val_loss_history.append(avg_val_loss)

        if epochs_no_improve >= early_stopping_patience:
            if verbose_steps > 0:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    if verbose_steps > 0:
        print(f"Training completed. Saving loss history to {out_files_dir}")
    save_loss_history(train_loss_history, val_loss_history, out_files_dir)


def save_model(
    model: torch.nn.Module, out_files_dir: str, model_name: str = "state_dict.pth"
) -> str:
    model_file_name = os.path.join(out_files_dir, model_name)
    torch.save(model.state_dict(), model_file_name)
    return model_file_name


def save_loss_history(
    train_loss_history: list[float],
    val_loss_history: list[float],
    out_files_dir: str,
    loss_history_file_name: str = "loss_history.json",
) -> None:
    loss_history = {
        "train_loss": train_loss_history,
        "val_loss": val_loss_history,
    }

    loss_history_file_name = os.path.join(out_files_dir, loss_history_file_name)
    with open(loss_history_file_name, "w") as f:
        json.dump(loss_history, f)
