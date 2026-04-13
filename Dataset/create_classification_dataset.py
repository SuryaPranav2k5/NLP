import pandas as pd
import re

df = pd.read_csv("mpvpn_industry_patrol_notes.csv")

# Function to extract additional features (for classical ML models, not for BERT)
def extract_features(note):
    # Normalize note once at the top
    note = note.lower()
    features = {}
    
    # Word count features
    tokens = note.split()
    features['word_count'] = len(tokens)
    features['char_count'] = len(note)
    
    # Presence of specific keywords (using word-boundary matching)
    features['has_repeat'] = 1 if re.search(r'\brepeat\b', note) else 0
    features['has_permit'] = 1 if re.search(r'\bpermit\b', note) else 0
    features['has_expired'] = 1 if re.search(r'\bexpired\b', note) else 0
    
    # Time-related features
    features['has_morning'] = 1 if re.search(r'\bmorning\b', note) else 0
    features['has_evening'] = 1 if re.search(r'\bevening\b', note) else 0
    features['has_night'] = 1 if re.search(r'\bnight\b', note) else 0
    
    # Location-related features
    features['has_zone'] = 1 if re.search(r'\bzone\b', note) else 0
    features['has_hospital'] = 1 if re.search(r'\bhospital\b', note) else 0
    features['has_school'] = 1 if re.search(r'\bschool\b', note) else 0
    
    # Vehicle-related features
    vehicle_types = ['car', 'bike', 'suv', 'truck', 'auto']
    features['has_vehicle'] = 1 if any(re.search(r'\b' + v + r'\b', note) for v in vehicle_types) else 0
    
    # Color-related features
    colors = ['white', 'black', 'red', 'blue', 'silver']
    features['has_color'] = 1 if any(re.search(r'\b' + c + r'\b', note) for c in colors) else 0
    
    return features

# Extract features for classical ML models (not for BERT)
enhanced_features = df['note'].apply(extract_features)
features_df = pd.DataFrame(enhanced_features.tolist())

# For BERT/Transformer models, we only need the raw text and labels
bert_clf_df = df[[
    "note",
    "violation_category",
    "violation_subtype",
    "severity"
]]

# For classical ML models, we can include engineered features
ml_clf_df = pd.concat([df[[
    "note",
    "violation_category",
    "violation_subtype",
    "severity",
    "zone_type",
    "patrol_shift",
    "repeat_flag",
    "permit_status",
    "weather",
    "confidence_level",
    "noise_level"
]], features_df], axis=1)

# Save both versions
bert_clf_df.to_csv("mpvpn_bert_classification_dataset.csv", index=False)
ml_clf_df.to_csv("mpvpn_ml_classification_dataset.csv", index=False)

print("BERT-ready classification dataset created:", bert_clf_df.shape)
print("ML-ready classification dataset created:", ml_clf_df.shape)
print("Engineered features added for ML dataset:", list(features_df.columns))