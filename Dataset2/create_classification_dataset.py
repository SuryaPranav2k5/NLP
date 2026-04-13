import pandas as pd
import re
import numpy as np

df = pd.read_csv("mpvpn_comprehensive_dataset.csv")

# =============== FEATURE EXTRACTION FOR CLASSICAL ML ===============

def extract_enhanced_features(note):
    """
    Extract comprehensive features for classical ML models
    These features are NOT needed for BERT/Transformer models
    """
    note_lower = note.lower()
    features = {}
    
    # === BASIC TEXT STATISTICS ===
    tokens = note_lower.split()
    features['word_count'] = len(tokens)
    features['char_count'] = len(note_lower)
    features['avg_word_length'] = np.mean([len(w) for w in tokens]) if tokens else 0
    features['unique_word_ratio'] = len(set(tokens)) / len(tokens) if tokens else 0
    
    # === VEHICLE-RELATED FEATURES ===
    # Vehicle types
    two_wheelers = ['motorcycle', 'scooter', 'moped', 'bike', 'vespa']
    three_wheelers = ['auto', 'rickshaw', 'tuk-tuk', 'e-rickshaw']
    four_wheelers = ['car', 'sedan', 'suv', 'hatchback', 'van', 'minivan']
    commercial = ['truck', 'lorry', 'bus', 'taxi', 'cab', 'delivery']
    luxury = ['luxury', 'sports car', 'limousine', 'premium']
    specialized = ['ambulance', 'police', 'fire truck', 'tow truck', 'garbage']
    
    features['has_two_wheeler'] = int(any(re.search(rf'\b{v}\b', note_lower) for v in two_wheelers))
    features['has_three_wheeler'] = int(any(re.search(rf'\b{v}\b', note_lower) for v in three_wheelers))
    features['has_four_wheeler'] = int(any(re.search(rf'\b{v}\b', note_lower) for v in four_wheelers))
    features['has_commercial_vehicle'] = int(any(re.search(rf'\b{v}\b', note_lower) for v in commercial))
    features['has_luxury_vehicle'] = int(any(re.search(rf'\b{v}\b', note_lower) for v in luxury))
    features['has_specialized_vehicle'] = int(any(re.search(rf'\b{v}\b', note_lower) for v in specialized))
    
    # Colors
    colors = ['white', 'black', 'red', 'blue', 'silver', 'grey', 'green', 'yellow', 'brown', 'orange']
    features['has_color'] = int(any(re.search(rf'\b{c}\b', note_lower) for c in colors))
    
    # License plate pattern
    features['has_license_plate'] = int(bool(re.search(r'\b[A-Z]{2}-\d{2}-[A-Z]\d{4}\b', note, re.IGNORECASE)))
    
    # === VIOLATION TYPE INDICATORS ===
    # Obstruction keywords
    obstruction_words = ['blocking', 'obstructing', 'obstructed', 'blocked', 'blk', 'blkg']
    features['has_obstruction'] = int(any(re.search(rf'\b{w}\b', note_lower) for w in obstruction_words))
    
    # Parking keywords
    parking_words = ['parked', 'parking', 'parkd', 'pk']
    features['has_parking'] = int(any(re.search(rf'\b{w}\b', note_lower) for w in parking_words))
    
    # Emergency-related
    emergency_words = ['fire', 'hydrant', 'emergency', 'exit', 'ambulance', 'emerg']
    features['has_emergency'] = int(any(re.search(rf'\b{w}\b', note_lower) for w in emergency_words))
    
    # Traffic flow
    traffic_words = ['double', 'wrong side', 'lane', 'flow', 'traffic']
    features['has_traffic_issue'] = int(any(w in note_lower for w in traffic_words))
    
    # Permit-related
    permit_words = ['permit', 'pass', 'expired', 'valid', 'unauthorized']
    features['has_permit_mention'] = int(any(re.search(rf'\b{w}\b', note_lower) for w in permit_words))
    
    # Safety hazards
    safety_words = ['blind spot', 'crossing', 'zebra', 'pedestrian', 'danger', 'accident']
    features['has_safety_concern'] = int(any(w in note_lower for w in safety_words))
    
    # === LOCATION-RELATED FEATURES ===
    # Zone types
    zones = ['residential', 'commercial', 'hospital', 'school', 'vip', 'industrial', 'heritage']
    features['has_zone_mention'] = int(any(re.search(rf'\b{z}\b', note_lower) for z in zones))
    features['has_hospital'] = int(bool(re.search(r'\bhospital\b', note_lower)))
    features['has_school'] = int(bool(re.search(r'\bschool\b', note_lower)))
    features['has_vip'] = int(bool(re.search(r'\bvip\b', note_lower)))
    
    # Specific locations
    road_words = ['road', 'street', 'highway', 'flyover', 'bridge', 'intersection']
    features['has_road_mention'] = int(any(re.search(rf'\b{w}\b', note_lower) for w in road_words))
    
    public_places = ['bus stop', 'metro', 'railway', 'station', 'terminal', 'park']
    features['has_public_place'] = int(any(w in note_lower for w in public_places))
    
    # === TIME-RELATED FEATURES ===
    time_words = ['morning', 'evening', 'night', 'afternoon', 'rush', 'peak', 'lunch']
    features['has_time_mention'] = int(any(re.search(rf'\b{w}\b', note_lower) for w in time_words))
    features['has_morning'] = int(bool(re.search(r'\bmorning\b', note_lower)))
    features['has_evening'] = int(bool(re.search(r'\bevening\b', note_lower)))
    features['has_night'] = int(bool(re.search(r'\bnight\b', note_lower)))
    features['has_rush_hour'] = int('rush' in note_lower or 'peak' in note_lower)
    
    # Duration mentions
    duration_words = ['hours', 'mins', 'overnight', 'all day', 'temporarily', 'hrs']
    features['has_duration'] = int(any(w in note_lower for w in duration_words))
    
    # === STATUS INDICATORS ===
    # Repeat violations
    repeat_words = ['repeat', 'multiple', 'habitual', 'again', 'known offender']
    features['has_repeat_indicator'] = int(any(w in note_lower for w in repeat_words))
    
    # Permit status
    features['has_expired'] = int(bool(re.search(r'\bexpired\b', note_lower)))
    features['has_invalid'] = int('invalid' in note_lower or 'wrong zone' in note_lower)
    
    # === SEVERITY INDICATORS ===
    # Aggravating factors
    aggravating = ['emergency', 'warning', 'refused', 'absent', 'locked', 'traffic jam', 'complaint']
    features['has_aggravating'] = int(any(w in note_lower for w in aggravating))
    
    # Mitigating factors
    mitigating = ['breakdown', 'cooperative', 'first offense', 'moved immediately', 'brief']
    features['has_mitigating'] = int(any(w in note_lower for w in mitigating))
    
    # === WEATHER/ENVIRONMENTAL ===
    weather_words = ['rain', 'fog', 'storm', 'mist', 'drizzle', 'hot', 'cold', 'windy']
    features['has_weather_mention'] = int(any(re.search(rf'\b{w}\b', note_lower) for w in weather_words))
    features['has_poor_weather'] = int(any(w in note_lower for w in ['rain', 'fog', 'storm']))
    
    visibility_words = ['visibility', 'view', 'obstructed view', 'poor visibility']
    features['has_visibility_issue'] = int(any(w in note_lower for w in visibility_words))
    
    # === PATROL/CONFIDENCE INDICATORS ===
    confidence_words = ['verified', 'confirmed', 'witnessed', 'captured', 'suspected', 'reported']
    features['has_confidence_indicator'] = int(any(re.search(rf'\b{w}\b', note_lower) for w in confidence_words))
    features['has_high_confidence'] = int(any(w in note_lower for w in ['verified', 'confirmed', 'witnessed', 'captured']))
    
    # === TEXT QUALITY/NOISE INDICATORS ===
    # Abbreviation count (proxy for noise level)
    abbr_patterns = [r'\bparkd\b', r'\bblk\b', r'\bnr\b', r'\bveh\b', r'\bobs\b', 
                     r'\bchkd\b', r'\bst\b', r'\bhrs\b', r'\bemerg\b']
    features['abbreviation_count'] = sum(bool(re.search(p, note_lower)) for p in abbr_patterns)
    
    # Punctuation count
    features['punctuation_count'] = sum(c in '.,;:!?' for c in note_lower)
    
    # Sentence count (approximate)
    features['sentence_count'] = note_lower.count('.') + note_lower.count('!') + note_lower.count('?') + 1
    
    return features

# =============== CREATE DATASETS ===============

print("Extracting features for classical ML models...")
enhanced_features = df['note'].apply(extract_enhanced_features)
features_df = pd.DataFrame(enhanced_features.tolist())

# === BERT/TRANSFORMER DATASET ===
# For BERT models, we only need raw text and labels
bert_clf_df = df[[
    "note",
    "violation_category",
    "violation_subtype",
    "severity"
]].copy()

# === CLASSICAL ML DATASET ===
# For classical ML, include all metadata and engineered features
ml_clf_df = pd.concat([
    df[[
        "note",
        "violation_category",
        "violation_subtype",
        "severity",
        "zone_type",
        "zone_description",
        "location_type",
        "location_description",
        "patrol_shift",
        "time_context",
        "duration",
        "repeat_flag",
        "permit_status",
        "weather",
        "visibility",
        "aggravating_factor",
        "mitigating_factor",
        "patrol_type",
        "confidence_level",
        "noise_level"
    ]],
    features_df
], axis=1)

# === MULTI-LABEL CLASSIFICATION DATASET ===
# Create binary labels for multi-label classification scenarios
multilabel_df = bert_clf_df.copy()

# Add binary indicators for different aspects
multilabel_df['is_obstruction'] = (df['violation_category'] == 'OBSTRUCTION').astype(int)
multilabel_df['is_traffic_flow'] = (df['violation_category'] == 'TRAFFIC_FLOW').astype(int)
multilabel_df['is_no_parking'] = (df['violation_category'] == 'NO_PARKING_ZONES').astype(int)
multilabel_df['is_permit_violation'] = (df['violation_category'] == 'PERMIT_VIOLATIONS').astype(int)
multilabel_df['is_restricted_area'] = (df['violation_category'] == 'RESTRICTED_AREAS').astype(int)
multilabel_df['is_safety_hazard'] = (df['violation_category'] == 'SAFETY_HAZARDS').astype(int)
multilabel_df['is_environmental'] = (df['violation_category'] == 'ENVIRONMENTAL').astype(int)
multilabel_df['is_meter_payment'] = (df['violation_category'] == 'METER_PAYMENT').astype(int)
multilabel_df['is_repeat_offense'] = (df['violation_category'] == 'REPEAT_OFFENSES').astype(int)

# Severity indicators
multilabel_df['is_critical'] = (df['severity'] == 'Critical').astype(int)
multilabel_df['is_high'] = (df['severity'] == 'High').astype(int)
multilabel_df['is_medium'] = (df['severity'] == 'Medium').astype(int)
multilabel_df['is_low'] = (df['severity'] == 'Low').astype(int)

# === SAVE DATASETS ===
bert_clf_df.to_csv("mpvpn_bert_classification_dataset.csv", index=False)
ml_clf_df.to_csv("mpvpn_ml_classification_dataset.csv", index=False)
multilabel_df.to_csv("mpvpn_multilabel_classification_dataset.csv", index=False)

# === STATISTICS ===
print("\n" + "="*60)
print("CLASSIFICATION DATASETS CREATED")
print("="*60)

print(f"\n✓ BERT Classification Dataset: {bert_clf_df.shape}")
print("  Columns: note, violation_category, violation_subtype, severity")
print("  Use for: BERT, RoBERTa, DistilBERT, or any transformer-based models")

print(f"\n✓ ML Classification Dataset: {ml_clf_df.shape}")
print(f"  Engineered features added: {len(features_df.columns)}")
print("  Use for: Random Forest, XGBoost, SVM, Logistic Regression, etc.")

print(f"\n✓ Multi-label Classification Dataset: {multilabel_df.shape}")
print("  Binary labels for: 9 violation categories + 4 severity levels")
print("  Use for: Multi-label classification tasks")

print("\n" + "="*60)
print("LABEL DISTRIBUTIONS")
print("="*60)

print("\nViolation Categories:")
print(bert_clf_df['violation_category'].value_counts())

print("\nViolation Subtypes (top 10):")
print(bert_clf_df['violation_subtype'].value_counts().head(10))

print("\nSeverity Levels:")
print(bert_clf_df['severity'].value_counts())

print("\n" + "="*60)
print("ENGINEERED FEATURES (Classical ML)")
print("="*60)
print("\nFeature categories:")
print("  - Text statistics (4 features)")
print("  - Vehicle-related (7 features)")
print("  - Violation type indicators (6 features)")
print("  - Location-related (6 features)")
print("  - Time-related (6 features)")
print("  - Status indicators (3 features)")
print("  - Severity indicators (2 features)")
print("  - Weather/environmental (3 features)")
print("  - Confidence indicators (2 features)")
print("  - Text quality/noise (3 features)")
print(f"\nTotal engineered features: {len(features_df.columns)}")
print(f"Feature names: {list(features_df.columns)}")

print("\n" + "="*60)
print("SAMPLE RECORDS")
print("="*60)
for i in range(3):
    row = bert_clf_df.iloc[i]
    print(f"\n{i+1}. Note: {row['note']}")
    print(f"   Category: {row['violation_category']}")
    print(f"   Subtype: {row['violation_subtype']}")
    print(f"   Severity: {row['severity']}")