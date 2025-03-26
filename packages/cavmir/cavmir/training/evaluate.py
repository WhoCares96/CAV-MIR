import json

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from cavmir.training.fit import get_samples_from_batch
from cavmir.training.network import CAVNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_cav_model(
    model: CAVNetwork,
    test_dataloader: DataLoader,
    true_label_name: str,
    loss_history_dir: str | None = None,
    device=DEVICE,
    plot_evaluation=False,
) -> dict:
    model = model.to(device)

    all_predicitions = []
    all_predicted_labels = []

    true_labels = []

    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, targets = get_samples_from_batch(batch, device)

            prediction = model.forward(inputs)
            prediction_labels = (prediction > 0.5).int()

            all_predicitions.extend(prediction.cpu().numpy())
            all_predicted_labels.extend(prediction_labels.cpu().numpy())
            true_labels.extend(targets.cpu().numpy())

    precision = precision_score(true_labels, all_predicted_labels, average="binary")
    recall = recall_score(true_labels, all_predicted_labels, average="binary")
    f1 = f1_score(true_labels, all_predicted_labels, average="binary")
    accuracy = accuracy_score(true_labels, all_predicted_labels)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
    }

    if plot_evaluation is False:
        return metrics

    cm = confusion_matrix(true_labels, all_predicted_labels)

    fpr, tpr, _ = roc_curve(true_labels, all_predicitions)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(true_labels, all_predicitions)

    if loss_history_dir:
        fig, ax = plt.subplots(1, 4, figsize=(21, 5))
    else:
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    fig.suptitle(f'CAV Model Evaluation, Concept: "{true_label_name}"')

    if loss_history_dir is not None:
        # Loss History
        loss_history = json.load(open(loss_history_dir))
        ax[0].plot(loss_history["train_loss"], label="train")
        ax[0].plot(loss_history["val_loss"], label="val")
        ax[0].set_title("Loss History")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")

    # Confusion Matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"not {true_label_name}", true_label_name],
        yticklabels=[f"not {true_label_name}", true_label_name],
        ax=ax[-3],
    )
    ax[-3].set_title("Confusion Matrix")
    ax[-3].set_xlabel("Predicted Label")
    ax[-3].set_ylabel("True Label")

    # ROC Curve
    ax[-2].plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
    ax[-2].plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax[-2].set_title("ROC Curve")
    ax[-2].set_xlabel("False Positive Rate")
    ax[-2].set_ylabel("True Positive Rate")
    ax[-2].legend(loc="lower right")

    # Precision-Recall Curve
    ax[-1].plot(recall, precision, color="green", lw=2)
    ax[-1].set_title("Precision-Recall Curve")
    ax[-1].set_xlabel("Recall")
    ax[-1].set_ylabel("Precision")

    plt.tight_layout()
    plt.show()

    return metrics
