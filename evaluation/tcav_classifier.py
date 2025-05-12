import torch, numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import torch.nn as nn
import torch.optim as optim


class EmbeddingDataset(Dataset):
    def __init__(self, df, embedding_col="embedding", target_col="genre"):
        self.embeddings = np.stack(df[embedding_col].values)
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(df[target_col].values)

    def __len__(self):          return len(self.embeddings)
    def __getitem__(self, idx): # returns one sample
        emb  = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        lab  = torch.tensor(self.labels[idx],     dtype=torch.long)
        return emb, lab


class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):       # x : [B, D]
        return self.fc(x)

    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def train_classifier(
        cls, df,
        embedding_col="embedding", target_col="genre",
        epochs=20, batch_size=64, learning_rate=1e-3,
        device="cpu"
    ):
        """
        Train a linear classifier and log:
            • epoch CE‑loss
            • first‑10 sample sanity‑check 
            • per‑genre F1 on the current train set  
        """

        '''
        print("genre classifiers gender value count:\n",
              df.groupby("genre")["gender"].value_counts(normalize=True), "\n")

        '''

        genre_counts = df[target_col].value_counts()      # pandas Series
        print("\n┌─ Training‑set size for genre classifier for TCAV per genre ───────────────────────────────")
        for genre, n in genre_counts.items():
            print(f"{genre:<25}: {n}")
        print("└───────────────────────────────────────────────────────────────\n")

        # ---- build dataset / dataloader -------------------------------------
        dataset   = EmbeddingDataset(df, embedding_col, target_col)
        dataloader= DataLoader(dataset, batch_size=batch_size, shuffle=True)

        input_dim   = dataset.embeddings.shape[1]
        num_classes = len(np.unique(dataset.labels))

        model = cls(input_dim, num_classes).to(device)
        optim_ = optim.Adam(model.parameters(), lr=learning_rate)
        crit   = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            # ---- training step ---------------------------------------------
            model.train()
            running_loss = 0.0
            for xb, yb in dataloader:
                xb, yb = xb.to(device), yb.to(device)

                optim_.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                optim_.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")

            # =================================================================
            # Mini‑batch sanity‑check 
            # =================================================================
            model.eval()
            with torch.no_grad():
                batch_emb, batch_lab = next(iter(dataloader))
                batch_emb = batch_emb.to(device)
                logits    = model(batch_emb)
                preds     = torch.argmax(logits, dim=1)
            '''
            print("\n╒═ Mini‑batch sanity‑check ═════════════════════════════════════")
            for i in range(min(10, len(batch_lab))):
                true_id  = batch_lab[i].item()
                pred_id  = preds[i].item()
                true_lbl = dataset.label_encoder.inverse_transform([true_id])[0]
                pred_lbl = dataset.label_encoder.inverse_transform([pred_id])[0]
                print(f"{i:2d} | true: {true_id:2d} ({true_lbl:<20})  "
                      f"pred: {pred_id:2d} ({pred_lbl})")
            print("╘═══════════════════════════════════════════════════════════════")
            '''
        if epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                all_logits, all_labels = [], []
                for emb_batch, lab_batch in dataloader:
                    emb_batch = emb_batch.to(device)
                    logits     = model(emb_batch)
                    all_logits.append(logits.cpu())
                    all_labels.append(lab_batch)

                y_true = torch.cat(all_labels).numpy()
                y_pred = torch.argmax(torch.cat(all_logits), dim=1).numpy()

                f1_per_class = f1_score(
                    y_true, y_pred, average=None, labels=np.arange(num_classes)
                )

                # pretty print
                print("\n┌─ Per‑genre F1 (training set, final epoch) ───────────────")
                for idx, f1 in enumerate(f1_per_class):
                    genre = dataset.label_encoder.inverse_transform([idx])[0]
                    print(f"{genre:<25}: {f1:>.3f}")
                print("└────────────────────────────────────────────────────────\n")


        return model, dataset
