import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertModel
from torch.optim import AdamW
from tqdm import tqdm
import ssl
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import numpy as np

# ---------------- SSL FIX (restricted networks) ----------------
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['CURL_CA_BUNDLE'] = ''

# ================= CONFIG =================
MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "mpvpn_ner_dataset.conll"
OUTPUT_DIR = "distilbert_mpvpn_ner"
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 6
LR = 3e-5
PATIENCE = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= LOAD CoNLL DATA =================
def load_conll(path):
    sentences, labels = [], []
    tokens, tags = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            # Sentence separator: empty line OR line with just comma
            if not line or line == ',':
                if tokens:
                    sentences.append(tokens)
                    labels.append(tags)
                    tokens, tags = [], []
            else:
                parts = line.split(',')
                if len(parts) == 2:
                    token, tag = parts
                    tokens.append(token)
                    tags.append(tag)
                # Skip malformed lines

    if tokens:
        sentences.append(tokens)
        labels.append(tags)

    return sentences, labels

sentences, ner_labels = load_conll(DATA_PATH)

# ================= DATA VALIDATION =================
print(f"📊 Dataset Statistics:")
print(f"   Total sentences: {len(sentences)}")
print(f"   Total tokens: {sum(len(s) for s in sentences)}")
print(f"   Avg tokens/sentence: {np.mean([len(s) for s in sentences]):.2f}")

# Check for empty sentences
assert all(len(s) > 0 for s in sentences), "Found empty sentences"
assert all(len(s) == len(l) for s, l in zip(sentences, ner_labels)), "Mismatch between tokens and labels"

# Check label consistency
all_tags = {tag for sent in ner_labels for tag in sent}
print(f"   Unique tags: {sorted(all_tags)}")
print(f"   Tag distribution: {dict(sorted([(tag, sum(1 for s in ner_labels for t in s if t == tag)) for tag in all_tags], key=lambda x: -x[1]))}")

# ================= LABEL MAP =================
unique_tags = sorted({tag for sent in ner_labels for tag in sent})
label2id = {tag: idx for idx, tag in enumerate(unique_tags)}
id2label = {idx: tag for tag, idx in label2id.items()}

with open(os.path.join(OUTPUT_DIR, "label_map.json"), "w") as f:
    json.dump(label2id, f, indent=4)

num_labels = len(label2id)

# ================= TOKENIZER =================
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

# ================= DATASET =================
class NERDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        tags = self.labels[idx]

        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_offsets_mapping=True
        )

        word_ids = encoding.word_ids()
        label_ids = []

        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                label_ids.append(label2id[tags[word_id]])
            else:
                label_ids.append(-100)
            prev_word_id = word_id

        encoding.pop("offset_mapping")

        return {
            "input_ids": torch.tensor(encoding["input_ids"]),
            "attention_mask": torch.tensor(encoding["attention_mask"]),
            "labels": torch.tensor(label_ids)
        }

# ================= SPLIT =================
split = int(0.85 * len(sentences))
train_data = NERDataset(sentences[:split], ner_labels[:split])
val_data = NERDataset(sentences[split:], ner_labels[split:])

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)

# ================= MODEL =================
class DistilBERT_NER(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(MODEL_NAME)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.view(-1, num_labels), labels.view(-1))

        return loss, logits

model = DistilBERT_NER(num_labels).to(DEVICE)

# ================= OPTIMIZER =================
optimizer = AdamW(model.parameters(), lr=LR)

# ================= EVALUATION FUNCTION =================
def evaluate_model(model, dataloader, device, id2label):
    """Evaluate model and return loss + NER metrics (precision, recall, F1)"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            loss, logits = model(input_ids, attention_mask, labels)
            total_loss += loss.item()
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Convert to label strings (ignore -100)
            for pred_seq, label_seq in zip(predictions, labels):
                pred_labels = []
                true_labels = []
                for p, l in zip(pred_seq.cpu().numpy(), label_seq.cpu().numpy()):
                    if l != -100:
                        pred_labels.append(id2label[p])
                        true_labels.append(id2label[l])
                
                if pred_labels:  # Only add non-empty sequences
                    all_predictions.append(pred_labels)
                    all_labels.append(true_labels)
    
    avg_loss = total_loss / len(dataloader)
    
    # Calculate NER metrics using seqeval
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    
    return avg_loss, precision, recall, f1

# ================= TRAINING =================
best_val_loss = float("inf")
patience_counter = 0

for epoch in range(EPOCHS):
    # ---- TRAIN ----
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        loss, _ = model(input_ids, attention_mask, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # ---- VALIDATE ----
    print(f"\n{'='*60}")
    print(f"Epoch {epoch+1}/{EPOCHS} - Validation")
    print(f"{'='*60}")
    
    val_loss, val_precision, val_recall, val_f1 = evaluate_model(
        model, val_loader, DEVICE, id2label
    )
    
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Val Metrics: Precision={val_precision:.4f}, Recall={val_recall:.4f}, F1={val_f1:.4f}")

    # ---- EARLY STOPPING ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_ner_model.pt"))
        print("✅ Best NER model saved")
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("🛑 Early stopping triggered")
            break

# ================= SAVE TOKENIZER & CONFIG =================
tokenizer.save_pretrained(OUTPUT_DIR)

training_config = {
    "model": MODEL_NAME,
    "max_length": MAX_LEN,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LR,
    "early_stopping": PATIENCE
}

with open(os.path.join(OUTPUT_DIR, "training_config.json"), "w") as f:
    json.dump(training_config, f, indent=4)

# ================= FINAL EVALUATION =================
print("\n" + "="*60)
print("FINAL EVALUATION ON VALIDATION SET")
print("="*60)

model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_ner_model.pt")))
final_loss, final_precision, final_recall, final_f1 = evaluate_model(
    model, val_loader, DEVICE, id2label
)

print(f"Best Model Performance:")
print(f"  Loss:      {final_loss:.4f}")
print(f"  Precision: {final_precision:.4f}")
print(f"  Recall:    {final_recall:.4f}")
print(f"  F1-Score:  {final_f1:.4f}")

# Generate detailed classification report
model.eval()
all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        
        _, logits = model(input_ids, attention_mask, labels)
        predictions = torch.argmax(logits, dim=-1)
        
        for pred_seq, label_seq in zip(predictions, labels):
            pred_labels = []
            true_labels = []
            for p, l in zip(pred_seq.cpu().numpy(), label_seq.cpu().numpy()):
                if l != -100:
                    pred_labels.append(id2label[p])
                    true_labels.append(id2label[l])
            
            if pred_labels:
                all_predictions.append(pred_labels)
                all_labels.append(true_labels)

print("\n" + "="*60)
print("DETAILED CLASSIFICATION REPORT")
print("="*60)
print(classification_report(all_labels, all_predictions))

print("\n🎉 Phase 3 NER training complete")
print("NER model saved to:", OUTPUT_DIR)
print(f"✅ Final F1-Score: {final_f1:.4f}")
