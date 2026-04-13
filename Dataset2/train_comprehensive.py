"""
Comprehensive Training Pipeline for Parking Violation Dataset
- DistilBERT for Classification (Category, Subtype, Severity)
- DistilBERT for Named Entity Recognition (NER)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                             f1_score, accuracy_score, precision_recall_fscore_support)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# ================================================================================
# PART 1: DISTILBERT CLASSIFICATION TRAINING
# ================================================================================

def train_distilbert_classifier():
    """
    Train DistilBERT classifier for violation category, subtype, and severity
    """
    print("\n" + "="*80)
    print("DISTILBERT CLASSIFICATION TRAINING")
    print("="*80)
    
    try:
        from transformers import (DistilBertTokenizer, DistilBertForSequenceClassification, 
                                  Trainer, TrainingArguments, EarlyStoppingCallback)
        import torch
        from torch.utils.data import Dataset
    except ImportError:
        print("⚠️  transformers or torch not installed. Install with:")
        print("   pip install transformers torch datasets accelerate")
        return
    
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Using device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\n📊 Loading BERT dataset...")
    df = pd.read_csv("/kaggle/input/datase/mpvpn_bert_classification_dataset.csv")
    print(f"Dataset shape: {df.shape}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"\n⚠️  Missing values found:\n{missing[missing > 0]}")
        df = df.dropna()
        print(f"After dropping: {df.shape}")
    
    print(f"\nSample records:")
    for i in range(2):
        print(f"\n{i+1}. Note: {df.iloc[i]['note'][:100]}...")
        print(f"   Category: {df.iloc[i]['violation_category']}")
        print(f"   Subtype: {df.iloc[i]['violation_subtype']}")
        print(f"   Severity: {df.iloc[i]['severity']}")
    
    # Train 3 separate models for: category, subtype, and severity
    tasks = {
        'category': 'violation_category',
        'subtype': 'violation_subtype', 
        'severity': 'severity'
    }
    
    results_summary = {}
    
    for task_name, label_col in tasks.items():
        print(f"\n{'='*80}")
        print(f"TRAINING TASK: {task_name.upper()}")
        print(f"{'='*80}")
        
        # Prepare labels
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(df[label_col])
        num_labels = len(label_encoder.classes_)
        id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
        label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
        
        print(f"\n📋 Task Info:")
        print(f"   Number of classes: {num_labels}")
        print(f"   Classes: {list(label_encoder.classes_)}")
        print(f"\n   Class distribution:")
        for cls, count in df[label_col].value_counts().items():
            print(f"      {cls}: {count} ({count/len(df)*100:.1f}%)")
        
        # Split data (80/10/10)
        X_train, X_temp, y_train, y_temp = train_test_split(
            df['note'].values, 
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, 
            y_temp,
            test_size=0.5,
            random_state=42,
            stratify=y_temp
        )
        
        print(f"\n📊 Data splits:")
        print(f"   Train: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
        print(f"   Val:   {len(X_val):,} samples ({len(X_val)/len(df)*100:.1f}%)")
        print(f"   Test:  {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")
        
        # Tokenizer
        print("\n🔧 Loading DistilBERT tokenizer...")
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Custom Dataset class
        class ParkingViolationDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length=128):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = str(self.texts[idx])
                label = self.labels[idx]
                
                encoding = self.tokenizer(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_attention_mask=True,
                    return_tensors='pt'
                )
                
                return {
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }
        
        # Create datasets
        print("\n📦 Creating datasets...")
        train_dataset = ParkingViolationDataset(X_train, y_train, tokenizer)
        val_dataset = ParkingViolationDataset(X_val, y_val, tokenizer)
        test_dataset = ParkingViolationDataset(X_test, y_test, tokenizer)
        
        # Load model
        print("\n🤖 Loading DistilBERT model...")
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id
        )
        
        # Training arguments
        output_dir = f'./distilbert_{task_name}_model'
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="no",
            seed=42,
            report_to="none"
        )
        
        # Compute metrics function
        def compute_metrics(pred):
            labels = pred.label_ids
            preds = pred.predictions.argmax(-1)
            
            # Calculate metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average='weighted', zero_division=0
            )
            acc = accuracy_score(labels, preds)
            
            return {
                'accuracy': acc,
                'f1_weighted': f1,
                'precision': precision,
                'recall': recall
            }
        
        # Create Trainer
        print("\n🏋️  Starting training...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        
        # Train
        train_result = trainer.train()
        
        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        test_results = trainer.evaluate(test_dataset)
        
        # Predictions for detailed metrics
        predictions = trainer.predict(test_dataset)
        y_pred = predictions.predictions.argmax(-1)
        
        # Print results
        print(f"\n{'='*80}")
        print(f"RESULTS FOR {task_name.upper()}")
        print(f"{'='*80}")
        print(f"\n📈 Test Metrics:")
        print(f"   Accuracy:  {test_results['eval_accuracy']:.4f}")
        print(f"   F1 Score:  {test_results['eval_f1_weighted']:.4f}")
        print(f"   Precision: {test_results['eval_precision']:.4f}")
        print(f"   Recall:    {test_results['eval_recall']:.4f}")
        
        # Classification report
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(
            y_test, 
            y_pred, 
            target_names=label_encoder.classes_,
            digits=4
        ))
        
        # Confusion Matrix
        print(f"\n🔲 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_
        )
        plt.title(f'Confusion Matrix - {task_name.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{task_name}.png', dpi=300, bbox_inches='tight')
        print(f"   Saved: confusion_matrix_{task_name}.png")
        plt.close()
        
        # Save model
        print(f"\n💾 Saving model to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save label encoder
        import pickle
        with open(f'{output_dir}/label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        print(f"   ✓ Model saved")
        print(f"   ✓ Tokenizer saved")
        print(f"   ✓ Label encoder saved")
        
        # Store results
        results_summary[task_name] = {
            'accuracy': test_results['eval_accuracy'],
            'f1_score': test_results['eval_f1_weighted'],
            'precision': test_results['eval_precision'],
            'recall': test_results['eval_recall'],
            'num_classes': num_labels,
            'model_path': output_dir
        }
        
        print(f"\n✅ {task_name.upper()} training complete!\n")
    
    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    for task_name, metrics in results_summary.items():
        print(f"\n{task_name.upper()}:")
        print(f"   Classes:   {metrics['num_classes']}")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   F1 Score:  {metrics['f1_score']:.4f}")
        print(f"   Model:     {metrics['model_path']}")
    
    return results_summary


# ================================================================================
# PART 2: DISTILBERT NER TRAINING
# ================================================================================

def train_distilbert_ner():
    """
    Train DistilBERT for Named Entity Recognition
    """
    print("\n" + "="*80)
    print("DISTILBERT NER TRAINING")
    print("="*80)
    
    try:
        from transformers import (DistilBertTokenizerFast, DistilBertForTokenClassification,
                                  Trainer, TrainingArguments, DataCollatorForTokenClassification,
                                  EarlyStoppingCallback)
        import torch
        from torch.utils.data import Dataset
        from seqeval.metrics import classification_report as seq_classification_report
        from seqeval.metrics import f1_score as seq_f1_score
        from seqeval.scheme import IOB2
    except ImportError:
        print("⚠️  Required libraries not installed. Install with:")
        print("   pip install transformers torch datasets seqeval")
        return
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🖥️  Using device: {device}")
    
    # Load NER data
    print("\n📊 Loading NER dataset...")
    
    # Read CoNLL format
    sentences = []
    sentence_tokens = []
    sentence_tags = []
    
    with open('/kaggle/input/datase/mpvpn_ner_dataset.conll', 'r') as f:
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
    
    # Add last sentence if exists
    if sentence_tokens:
        sentences.append({
            'tokens': sentence_tokens,
            'tags': sentence_tags
        })
    
    print(f"   Total sentences: {len(sentences):,}")
    
    # Get unique tags
    all_tags = set()
    for sent in sentences:
        all_tags.update(sent['tags'])
    
    # Create tag mappings
    unique_tags = sorted(list(all_tags))
    tag2id = {tag: i for i, tag in enumerate(unique_tags)}
    id2tag = {i: tag for tag, i in tag2id.items()}
    num_labels = len(unique_tags)
    
    print(f"   Unique tags: {num_labels}")
    print(f"   Tags: {unique_tags}")
    
    # Count entities
    entity_counts = {}
    for sent in sentences:
        for tag in sent['tags']:
            if tag.startswith('B-'):
                entity_type = tag[2:]
                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    print(f"\n   Entity distribution:")
    for entity, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"      {entity}: {count:,}")
    
    # Split data
    train_size = int(0.8 * len(sentences))
    val_size = int(0.1 * len(sentences))
    
    train_sentences = sentences[:train_size]
    val_sentences = sentences[train_size:train_size+val_size]
    test_sentences = sentences[train_size+val_size:]
    
    print(f"\n📊 Data splits:")
    print(f"   Train: {len(train_sentences):,} sentences")
    print(f"   Val:   {len(val_sentences):,} sentences")
    print(f"   Test:  {len(test_sentences):,} sentences")
    
    # Load tokenizer
    print("\n🔧 Loading DistilBERT tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    
    # Tokenize and align labels
    def tokenize_and_align_labels(examples, max_length=128):
        tokenized_inputs = tokenizer(
            examples['tokens'],
            truncation=True,
            is_split_into_words=True,
            max_length=max_length,
            padding='max_length'
        )
        
        labels = []
        for i, label in enumerate(examples['tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(tag2id[label[word_idx]])
                else:
                    # For subword tokens, use -100 or same label
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    # Prepare datasets
    print("\n📦 Preparing datasets...")
    
    def prepare_dataset(sentences_list):
        tokens_list = [s['tokens'] for s in sentences_list]
        tags_list = [s['tags'] for s in sentences_list]
        
        dataset_dict = {
            'tokens': tokens_list,
            'tags': tags_list
        }
        
        return tokenize_and_align_labels(dataset_dict)
    
    train_encodings = prepare_dataset(train_sentences)
    val_encodings = prepare_dataset(val_sentences)
    test_encodings = prepare_dataset(test_sentences)
    
    # Create PyTorch datasets
    class NERDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings
        
        def __len__(self):
            return len(self.encodings['input_ids'])
        
        def __getitem__(self, idx):
            return {
                'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
                'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
                'labels': torch.tensor(self.encodings['labels'][idx])
            }
    
    train_dataset = NERDataset(train_encodings)
    val_dataset = NERDataset(val_encodings)
    test_dataset = NERDataset(test_encodings)
    
    # Load model
    print("\n🤖 Loading DistilBERT for token classification...")
    model = DistilBertForTokenClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_labels,
        id2label=id2tag,
        label2id=tag2id
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Compute metrics
    def compute_metrics(pred):
        predictions, labels = pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        # Calculate metrics using seqeval
        results = {
            'f1': seq_f1_score(true_labels, true_predictions, scheme=IOB2),
        }
        
        return results
    
    # Training arguments
    output_dir = './distilbert_ner_model'
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="no",
        seed=42,
        report_to="none"
    )
    
    # Create Trainer
    print("\n🏋️  Starting NER training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train
    train_result = trainer.train()
    
    # Evaluate on test set
    print("\n📊 Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    
    # Get predictions for detailed analysis
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=2)
    
    # Convert to label strings
    true_predictions = [
        [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, predictions.label_ids)
    ]
    true_labels = [
        [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, predictions.label_ids)
    ]
    
    # Print results
    print(f"\n{'='*80}")
    print("NER RESULTS")
    print(f"{'='*80}")
    print(f"\n📈 Test Metrics:")
    print(f"   F1 Score: {test_results['eval_f1']:.4f}")
    
    print(f"\n📋 Detailed Classification Report:")
    print(seq_classification_report(true_labels, true_predictions, scheme=IOB2, digits=4))
    
    # Save model
    print(f"\n💾 Saving NER model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save tag mappings
    import pickle
    with open(f'{output_dir}/tag_mappings.pkl', 'wb') as f:
        pickle.dump({'tag2id': tag2id, 'id2tag': id2tag}, f)
    
    print(f"   ✓ Model saved")
    print(f"   ✓ Tokenizer saved")
    print(f"   ✓ Tag mappings saved")
    
    # Show examples
    print(f"\n🔍 Sample Predictions:")
    for i in range(min(3, len(test_sentences))):
        sent = test_sentences[i]
        tokens = sent['tokens']
        true = sent['tags']
        
        # Get prediction for this sentence
        pred = true_predictions[i] if i < len(true_predictions) else []
        
        print(f"\n{i+1}. Sentence: {' '.join(tokens)}")
        print(f"   Entities found:")
        
        for j, (token, true_tag, pred_tag) in enumerate(zip(tokens, true, pred)):
            if true_tag != 'O' or pred_tag != 'O':
                match = "✓" if true_tag == pred_tag else "✗"
                print(f"      {match} {token:.<20} True: {true_tag:.<25} Pred: {pred_tag}")
    
    print(f"\n✅ NER training complete!")
    
    return {
        'f1_score': test_results['eval_f1'],
        'num_labels': num_labels,
        'model_path': output_dir
    }


# ================================================================================
# INFERENCE FUNCTIONS
# ================================================================================

def predict_classification(text, task='category'):
    """
    Make predictions using trained classification model
    
    Args:
        text: Input parking violation note
        task: 'category', 'subtype', or 'severity'
    """
    try:
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        import torch
        import pickle
    except ImportError:
        print("⚠️  transformers not installed")
        return None
    
    model_path = f'./distilbert_{task}_model'
    
    # Load model, tokenizer, and label encoder
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    
    with open(f'{model_path}/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = predictions[0][predicted_class].item()
    
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]
    
    return {
        'prediction': predicted_label,
        'confidence': confidence,
        'all_probabilities': {
            label_encoder.classes_[i]: predictions[0][i].item() 
            for i in range(len(label_encoder.classes_))
        }
    }


def predict_ner(text):
    """
    Extract entities from parking violation note using trained NER model
    
    Args:
        text: Input parking violation note
    """
    try:
        from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification
        import torch
        import pickle
    except ImportError:
        print("⚠️  transformers not installed")
        return None
    
    model_path = './distilbert_ner_model'
    
    # Load model, tokenizer, and tag mappings
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    model = DistilBertForTokenClassification.from_pretrained(model_path)
    
    with open(f'{model_path}/tag_mappings.pkl', 'rb') as f:
        mappings = pickle.load(f)
        id2tag = mappings['id2tag']
    
    # Tokenize
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128,
        return_offsets_mapping=True
    )
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    
    # Convert to labels
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    predicted_labels = [id2tag[pred.item()] for pred in predictions[0]]
    
    # Extract entities
    entities = []
    current_entity = None
    
    for token, label in zip(tokens, predicted_labels):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
        
        if label.startswith('B-'):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                'entity': label[2:],
                'text': token.replace('##', ''),
                'confidence': 1.0
            }
        elif label.startswith('I-') and current_entity:
            current_entity['text'] += ' ' + token.replace('##', '')
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    
    if current_entity:
        entities.append(current_entity)
    
    return entities


# ================================================================================
# MAIN EXECUTION
# ================================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPREHENSIVE PARKING VIOLATION TRAINING PIPELINE")
    print("="*80)
    
    import sys
    
    print("\nSelect training mode:")
    print("1. Train Classification Models (Category, Subtype, Severity)")
    print("2. Train NER Model")
    print("3. Train Both")
    print("4. Test Inference")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        results = train_distilbert_classifier()
    
    elif choice == '2':
        results = train_distilbert_ner()
    
    elif choice == '3':
        print("\n🚀 Training all models...\n")
        clf_results = train_distilbert_classifier()
        ner_results = train_distilbert_ner()
        
        print("\n" + "="*80)
        print("ALL TRAINING COMPLETE")
        print("="*80)
        
        if clf_results:
            print("\nClassification Models:")
            for task, metrics in clf_results.items():
                print(f"  {task}: F1={metrics['f1_score']:.4f}")
        else:
            print("\n⚠️  Classification training failed or was skipped")
        
        if ner_results:
            print(f"\nNER Model:")
            print(f"  F1={ner_results['f1_score']:.4f}")
        else:
            print("\n⚠️  NER training failed or was skipped")
    
    elif choice == '4':
        print("\n🔮 Testing Inference...\n")
        
        test_note = "during morning routine patrol observed white honda car blocking fire hydrant at main road in hospital zone. repeat violation, permit expired"
        
        print(f"Test Note: {test_note}\n")
        
        print("="*80)
        print("CLASSIFICATION PREDICTIONS")
        print("="*80)
        
        for task in ['category', 'subtype', 'severity']:
            result = predict_classification(test_note, task=task)
            if result:
                print(f"\n{task.upper()}:")
                print(f"  Prediction: {result['prediction']}")
                print(f"  Confidence: {result['confidence']:.4f}")
        
        print("\n" + "="*80)
        print("NER PREDICTIONS")
        print("="*80)
        
        entities = predict_ner(test_note)
        if entities:
            print(f"\nFound {len(entities)} entities:")
            for ent in entities:
                print(f"  • {ent['text']:.<30} [{ent['entity']}]")
        else:
            print("No entities found")
    
    else:
        print("Invalid choice!")
    
    print("\n✅ Done!")