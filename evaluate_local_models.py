import os
import sys
import torch
import pandas as pd
import numpy as np
import pickle
import joblib
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import classification_report as seq_classification_report
from seqeval.scheme import IOB2

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = "Dataset2"
BATCH_SIZE = 32

# Argument parsing
RUN_FULL = "--full" in sys.argv

print(f"Using device: {DEVICE}", flush=True)
if RUN_FULL:
    print("Running FULL evaluation mode on the entire test split.", flush=True)
else:
    print("Running QUICK evaluation mode (subset of test split). Run with '--full' for complete test set.", flush=True)

# ================================================================================
# PART 1: CLASSIFICATION METRICS
# ================================================================================

def evaluate_classification_tasks():
    csv_path = os.path.join(DATASET_DIR, "mpvpn_bert_classification_dataset.csv")
    if not os.path.exists(csv_path):
        print(f"⚠️ Classification dataset not found at {csv_path}. Skipping.", flush=True)
        return {}
        
    print(f"\nLoading classification dataset from {csv_path}...", flush=True)
    df = pd.read_csv(csv_path)
    
    tasks = {
        'category': {
            'dir': os.path.join(DATASET_DIR, "distilbert_category_model"),
            'col': 'violation_category'
        },
        'subtype': {
            'dir': os.path.join(DATASET_DIR, "distilbert_subtype_model"),
            'col': 'violation_subtype'
        },
        'severity': {
            'dir': os.path.join(DATASET_DIR, "distilbert_severity_model"),
            'col': 'severity'
        }
    }
    
    results = {}
    
    for task_name, info in tasks.items():
        model_dir = info['dir']
        label_col = info['col']
        
        if not os.path.exists(model_dir):
            print(f"⚠️ Model directory {model_dir} not found. Skipping {task_name}.", flush=True)
            continue
            
        print(f"\n--- Evaluating {task_name.upper()} ---", flush=True)
        
        # Load encoder and labels
        encoder_path = os.path.join(model_dir, "label_encoder.pkl")
        label_encoder = joblib.load(encoder_path)
        labels = label_encoder.transform(df[label_col])
        
        # Exact same split as training: 80% train, 10% val, 10% test
        _, X_temp, _, y_temp = train_test_split(
            df['note'].values, 
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
        
        _, X_test, _, y_test = train_test_split(
            X_temp, 
            y_temp,
            test_size=0.5,
            random_state=42,
            stratify=y_temp
        )
        
        # Slice for quick eval if not run full
        if not RUN_FULL:
            eval_size = min(500, len(X_test))
            print(f"Subset: evaluating {eval_size} samples (out of {len(X_test)} total test split)", flush=True)
            X_eval = X_test[:eval_size]
            y_eval = y_test[:eval_size]
        else:
            print(f"Evaluating all {len(X_test)} samples of test split", flush=True)
            X_eval = X_test
            y_eval = y_test
            
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(DEVICE)
        model.eval()
        
        # Custom batch dataset
        class EvalDataset(Dataset):
            def __init__(self, texts, tokenizer, max_length=128):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_length = max_length
                
            def __len__(self):
                return len(self.texts)
                
            def __getitem__(self, idx):
                text = str(self.texts[idx])
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten()
                }
        
        eval_dataset = EvalDataset(X_eval, tokenizer)
        dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        all_preds = []
        total_batches = len(dataloader)
        
        with torch.no_grad():
            for idx, batch in enumerate(dataloader):
                if idx % 5 == 0:
                    print(f"  Batch {idx + 1}/{total_batches}...", flush=True)
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
                all_preds.extend(preds)
                
        y_eval = y_eval.tolist()
        
        # Compute metrics
        acc = accuracy_score(y_eval, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_eval, all_preds, average='weighted', zero_division=0
        )
        
        print(f"Accuracy:  {acc:.4f}", flush=True)
        print(f"Precision: {precision:.4f}", flush=True)
        print(f"Recall:    {recall:.4f}", flush=True)
        print(f"F1-Score:  {f1:.4f} (Weighted)", flush=True)
        
        results[task_name] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_weighted': f1
        }
        
    return results

# ================================================================================
# PART 2: NER METRICS
# ================================================================================

def evaluate_ner():
    ner_path = os.path.join(DATASET_DIR, "mpvpn_ner_dataset.conll")
    model_dir = os.path.join(DATASET_DIR, "distilbert_ner_model")
    
    if not os.path.exists(ner_path) or not os.path.exists(model_dir):
        print(f"⚠️ NER dataset or model not found. Skipping.", flush=True)
        return {}
        
    print("\n--- Evaluating NER model ---", flush=True)
    
    # Read sentences from CoNLL
    sentences = []
    sentence_tokens = []
    sentence_tags = []
    
    with open(ner_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if sentence_tokens:
                    sentences.append({
                        'tokens': sentence_tokens,
                        'tags': sentence_tags
                    })
                    sentence_tokens = []
                    sentence_tags = []
            else:
                parts = line.split('\t')
                if len(parts) == 2:
                    token, tag = parts
                    sentence_tokens.append(token)
                    sentence_tags.append(tag)
                    
    if sentence_tokens:
        sentences.append({
            'tokens': sentence_tokens,
            'tags': sentence_tags
        })
        
    print(f"Total parsed sentences: {len(sentences)}", flush=True)
    
    # Exact same split as training: 80% train, 10% val, 10% test
    train_size = int(0.8 * len(sentences))
    val_size = int(0.1 * len(sentences))
    test_sentences = sentences[train_size+val_size:]
    
    if not RUN_FULL:
        eval_size = min(200, len(test_sentences))
        print(f"Subset: evaluating {eval_size} sentences (out of {len(test_sentences)} total test split)", flush=True)
        eval_sentences = test_sentences[:eval_size]
    else:
        print(f"Evaluating all {len(test_sentences)} sentences of test split", flush=True)
        eval_sentences = test_sentences
        
    # Load model and tag mappings
    with open(os.path.join(model_dir, "tag_mappings.pkl"), "rb") as f:
        tag_mappings = joblib.load(f)
        tag2id = tag_mappings['tag2id']
        id2tag = tag_mappings['id2tag']
        
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir).to(DEVICE)
    model.eval()
    
    # Batch predict
    true_labels = []
    pred_labels = []
    total_sents = len(eval_sentences)
    
    for idx, item in enumerate(eval_sentences):
        if idx % 50 == 0:
            print(f"  Sentence {idx + 1}/{total_sents}...", flush=True)
        tokens = item['tokens']
        tags = item['tags']
        
        encoding = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
            
        word_ids = encoding.word_ids()
        
        sent_preds = []
        sent_trues = []
        prev_word_idx = None
        
        for i, word_idx in enumerate(word_ids):
            if word_idx is None or word_idx == prev_word_idx:
                continue
            pred_id = predictions[i]
            label = id2tag.get(pred_id, id2tag.get(str(pred_id), "O"))
            sent_preds.append(label)
            sent_trues.append(tags[word_idx])
            prev_word_idx = word_idx
            
        pred_labels.append(sent_preds)
        true_labels.append(sent_trues)
        
    # Compute metrics using seqeval
    f1 = seq_f1_score(true_labels, pred_labels, scheme=IOB2)
    print(f"NER Entity-Level F1-Score: {f1:.4f}", flush=True)
    
    return {'f1': f1}

# ================================================================================
# MAIN
# ================================================================================

if __name__ == "__main__":
    print("================================================================================", flush=True)
    clf_res = evaluate_classification_tasks()
    ner_res = evaluate_ner()
    
    print("\n================================================================================", flush=True)
    print("EVALUATION SUMMARY", flush=True)
    print("================================================================================", flush=True)
    if clf_res:
        for task, metrics in clf_res.items():
            print(f"{task.upper():<10} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_weighted']:.4f}", flush=True)
    if ner_res:
        print(f"NER        | F1-Score (Entity-Level): {ner_res['f1']:.4f}", flush=True)
    print("================================================================================", flush=True)
