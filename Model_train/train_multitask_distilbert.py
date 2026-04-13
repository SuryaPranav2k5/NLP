import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from torch.optim import AdamW
import torch.nn as nn
from tqdm import tqdm
import joblib
import json
import ssl

# Bypass SSL verification for Hugging Face downloads (corporate firewall workaround)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''

# ================= CONFIG =================
MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "mpvpn_bert_classification_dataset.csv"
OUTPUT_DIR = "distilbert_mpvpn_multitask"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 6
LR = 2e-5
PATIENCE = 2              # Early stopping patience
CAT_LOSS_WEIGHT = 1.0
SEV_LOSS_WEIGHT = 0.6     # Less weight to easier task
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save training configuration for reproducibility
training_config = {
    "model_name": MODEL_NAME,
    "max_length": MAX_LEN,
    "batch_size": BATCH_SIZE,
    "learning_rate": LR,
    "epochs": EPOCHS,
    "train_val_split": 0.85,
    "category_loss_weight": CAT_LOSS_WEIGHT,
    "severity_loss_weight": SEV_LOSS_WEIGHT,
    "early_stopping_patience": PATIENCE
}

with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
    json.dump(training_config, f, indent=4)

# ================= LOAD DATA =================
df = pd.read_csv(DATA_PATH)

cat_encoder = LabelEncoder()
sev_encoder = LabelEncoder()

df["cat_label"] = cat_encoder.fit_transform(df["violation_category"])
df["sev_label"] = sev_encoder.fit_transform(df["severity"])

train_df, val_df = train_test_split(
    df,
    test_size=0.15,
    stratify=df["cat_label"],
    random_state=42
)

# ================= TOKENIZER =================
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

class MPVNDataset(Dataset):
    def __init__(self, texts, cat_labels, sev_labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_LEN
        )
        self.cat_labels = cat_labels
        self.sev_labels = sev_labels

    def __len__(self):
        return len(self.cat_labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["cat_labels"] = torch.tensor(self.cat_labels[idx])
        item["sev_labels"] = torch.tensor(self.sev_labels[idx])
        return item

train_loader = DataLoader(
    MPVNDataset(
        train_df["note"].tolist(),
        train_df["cat_label"].tolist(),
        train_df["sev_label"].tolist()
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    MPVNDataset(
        val_df["note"].tolist(),
        val_df["cat_label"].tolist(),
        val_df["sev_label"].tolist()
    ),
    batch_size=BATCH_SIZE
)

# ================= MODEL =================
class MultiTaskDistilBERT(nn.Module):
    def __init__(self, num_cat, num_sev):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.3)
        self.cat_head = nn.Linear(self.bert.config.hidden_size, num_cat)
        self.sev_head = nn.Linear(self.bert.config.hidden_size, num_sev)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        return self.cat_head(pooled), self.sev_head(pooled)

model = MultiTaskDistilBERT(
    num_cat=len(cat_encoder.classes_),
    num_sev=len(sev_encoder.classes_)
).to(DEVICE)

# ================= OPTIMIZER =================
optimizer = AdamW(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

# ================= TRAINING LOOP =================
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):
    # -------- TRAIN --------
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        cat_labels = batch["cat_labels"].to(DEVICE)
        sev_labels = batch["sev_labels"].to(DEVICE)

        cat_logits, sev_logits = model(input_ids, attention_mask)

        loss_cat = loss_fn(cat_logits, cat_labels)
        loss_sev = loss_fn(sev_logits, sev_labels)

        loss = CAT_LOSS_WEIGHT * loss_cat + SEV_LOSS_WEIGHT * loss_sev
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -------- VALIDATE --------
    model.eval()
    val_loss = 0.0
    
    # Initialize validation accuracy counters
    val_cat_correct = 0
    val_cat_total = 0
    val_sev_correct = 0
    val_sev_total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            cat_labels = batch["cat_labels"].to(DEVICE)
            sev_labels = batch["sev_labels"].to(DEVICE)

            cat_logits, sev_logits = model(input_ids, attention_mask)

            loss_cat = loss_fn(cat_logits, cat_labels)
            loss_sev = loss_fn(sev_logits, sev_labels)
            loss = CAT_LOSS_WEIGHT * loss_cat + SEV_LOSS_WEIGHT * loss_sev

            val_loss += loss.item()
            
            # Calculate predictions for accuracy
            cat_preds = torch.argmax(cat_logits, dim=1)
            sev_preds = torch.argmax(sev_logits, dim=1)
            
            # Update accuracy counters
            val_cat_correct += (cat_preds == cat_labels).sum().item()
            val_cat_total += cat_labels.size(0)
            
            val_sev_correct += (sev_preds == sev_labels).sum().item()
            val_sev_total += sev_labels.size(0)

    val_loss /= len(val_loader)
    
    # Calculate validation accuracies
    val_cat_acc = val_cat_correct / val_cat_total
    val_sev_acc = val_sev_correct / val_sev_total

    print(f"\nEpoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Validation Accuracy | Category: {val_cat_acc:.4f}, Severity: {val_sev_acc:.4f}")

    # -------- EARLY STOPPING --------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
        print("✅ Best model saved")

    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("🛑 Early stopping triggered")
            break

# ================= SAVE METADATA =================
tokenizer.save_pretrained(OUTPUT_DIR)
joblib.dump(cat_encoder, os.path.join(OUTPUT_DIR, "category_encoder.pkl"))
joblib.dump(sev_encoder, os.path.join(OUTPUT_DIR, "severity_encoder.pkl"))

print("\n🎉 Training complete")
print("Model + tokenizer + encoders saved to:", OUTPUT_DIR)

# Reload best model for inference safety
best_model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
model.to(DEVICE)
model.eval()

print("✅ Best model reloaded and ready for inference")
