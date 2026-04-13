import torch
import json
import joblib
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSequenceClassification
import torch.nn as nn

# ================= DEVICE =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= PATHS =================
DATASET_DIR = "Dataset2"
NER_DIR = f"{DATASET_DIR}/distilbert_ner_model"
CAT_DIR = f"{DATASET_DIR}/distilbert_category_model"
SEV_DIR = f"{DATASET_DIR}/distilbert_severity_model"
SUB_DIR = f"{DATASET_DIR}/distilbert_subtype_model"

# ================= LOAD TOKENIZERS =================
# We assume the tokenizer vocab is consistent, but we load from specific dirs to be safe
ner_tokenizer = AutoTokenizer.from_pretrained(NER_DIR)
cat_tokenizer = AutoTokenizer.from_pretrained(CAT_DIR)

# ================= LOAD LABEL MAPS & ENCODERS =================
# NER Label Map
with open(f"{NER_DIR}/tag_mappings.pkl", "rb") as f:
    tag_mappings = joblib.load(f)
    if "id2tag" in tag_mappings:
        id2label = tag_mappings["id2tag"]
    else:
        id2label = {v: k for k, v in tag_mappings["tag2id"].items()}

# Classification Encoders
category_encoder = joblib.load(f"{CAT_DIR}/label_encoder.pkl")
severity_encoder = joblib.load(f"{SEV_DIR}/label_encoder.pkl")
subtype_encoder = joblib.load(f"{SUB_DIR}/label_encoder.pkl")

# ================= LOAD MODELS =================
# NER Model
ner_model = AutoModelForTokenClassification.from_pretrained(NER_DIR)
ner_model.to(DEVICE)
ner_model.eval()

# Classification Models
cat_model = AutoModelForSequenceClassification.from_pretrained(CAT_DIR)
cat_model.to(DEVICE)
cat_model.eval()

sev_model = AutoModelForSequenceClassification.from_pretrained(SEV_DIR)
sev_model.to(DEVICE)
sev_model.eval()

sub_model = AutoModelForSequenceClassification.from_pretrained(SUB_DIR)
sub_model.to(DEVICE)
sub_model.eval()

# ================= ENTITY RECONSTRUCTION =================
def reconstruct_entities(tokens, labels):
    entities = {}
    current_entity = None
    current_tokens = []
    
    # Common words to filter out when they appear as standalone entities
    stopwords = {'a', 'an', 'the', 'is', 'was', 'were', 'been', 'be', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}

    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current_entity and current_tokens:
                entity_text = " ".join(current_tokens)
                # Only add if not a single stopword
                if len(current_tokens) > 1 or current_tokens[0] not in stopwords:
                    entities.setdefault(current_entity, []).append(entity_text)
            current_entity = label[2:]
            current_tokens = [token]
        elif label.startswith("I-") and current_entity:
            current_tokens.append(token)
        else:
            if current_entity and current_tokens:
                entity_text = " ".join(current_tokens)
                # Only add if not a single stopword
                if len(current_tokens) > 1 or current_tokens[0] not in stopwords:
                    entities.setdefault(current_entity, []).append(entity_text)
                current_entity = None
                current_tokens = []

    if current_entity and current_tokens:
        entity_text = " ".join(current_tokens)
        # Only add if not a single stopword
        if len(current_tokens) > 1 or current_tokens[0] not in stopwords:
            entities.setdefault(current_entity, []).append(entity_text)

    return entities

# ================= NER INFERENCE =================
def ner_inference(text):
    words = re.findall(r"\b\w+\b", text.lower())

    encoding = ner_tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = ner_model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    predictions = torch.argmax(logits, dim=-1)[0].cpu().numpy()
    word_ids = encoding.word_ids()

    final_labels = []
    final_tokens = []
    prev_word_id = None

    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == prev_word_id:
            continue
        final_tokens.append(words[word_id])
        # Ensure we don't crash if prediction is out of bounds (though it shouldn't be)
        # and convert using id2label
        label_id = predictions[idx]
        # id2label might have string keys if loaded from JSON, or int if from pkl, handle gracefully
        label = id2label.get(label_id, id2label.get(str(label_id), "O"))
        final_labels.append(label)
        prev_word_id = word_id

    return reconstruct_entities(final_tokens, final_labels)

# ================= CLASSIFICATION INFERENCE =================
def classification_inference(text):
    encoding = cat_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        cat_out = cat_model(input_ids, attention_mask=attention_mask)
        sev_out = sev_model(input_ids, attention_mask=attention_mask)
        sub_out = sub_model(input_ids, attention_mask=attention_mask)

    cat_pred = torch.argmax(cat_out.logits, dim=1).cpu().numpy()
    sev_pred = torch.argmax(sev_out.logits, dim=1).cpu().numpy()
    sub_pred = torch.argmax(sub_out.logits, dim=1).cpu().numpy()

    category = category_encoder.inverse_transform(cat_pred)[0]
    severity = severity_encoder.inverse_transform(sev_pred)[0]
    subtype = subtype_encoder.inverse_transform(sub_pred)[0]
    


    return {
        "violation_category": category,
        "severity": severity,
        "violation_subtype": subtype
    }

# ================= FULL PIPELINE =================
def analyze_patrol_note(text):
    entities = ner_inference(text)
    classification = classification_inference(text)

    return {
        "input_text": text,
        "entities": entities,
        "violation_category": classification["violation_category"],
        "severity": classification["severity"],
        "violation_subtype": classification["violation_subtype"]
    }

# ================= TEST =================
if __name__ == "__main__":
    sample_text = "black suv parked near fire hydrant at main road during night patrol"
    result = analyze_patrol_note(sample_text)

    print("\n🔍 ANALYSIS RESULT")
    print(json.dumps(result, indent=4))
