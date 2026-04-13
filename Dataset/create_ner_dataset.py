import pandas as pd
import re

df = pd.read_csv("mpvpn_industry_patrol_notes.csv")

# Enhanced entity lists
VEHICLES = ["car", "bike", "suv", "truck", "auto"]
COLORS = ["white", "black", "red", "blue", "silver"]

# Zone types for better recognition
ZONE_TYPES = ["residential", "commercial", "hospital", "school", "vip"]

# Locations for better recognition
LOCATIONS = ["main road", "market area", "hostel gate", "bus stop", "block a"]

# Reverse abbreviation mapping for NER matching
ABBR_REVERSE = {
    "hydrnt": "hydrant",
    "parkd": "parked",
    "blk": "blocking",
    "nr": "near",
    "st": "street",
    "clng": "cleaning"
}

# Multi-word entities for better recognition
MULTI_WORD_ENTITIES = {
    "fire hydrant": "B-VIOLATION_OBJECT",
    "emergency exit": "B-VIOLATION_OBJECT",
    "pedestrian path": "B-VIOLATION_OBJECT",
    "double parking": "B-VIOLATION_OBJECT",
    "wrong side parking": "B-VIOLATION_OBJECT",
    "street cleaning": "B-VIOLATION_OBJECT",
    "no parking anytime": "B-VIOLATION_OBJECT",
    "expired permit": "B-VIOLATION_OBJECT",
    "hospital zone": "B-ZONE_TYPE",
    "school zone": "B-ZONE_TYPE",
    "residential zone": "B-ZONE_TYPE",
    "commercial zone": "B-ZONE_TYPE",
    "vip zone": "B-ZONE_TYPE",
    "morning patrol": "B-TIME_RANGE",
    "evening patrol": "B-TIME_RANGE",
    "night patrol": "B-TIME_RANGE",
    "repeat violation": "B-REPEAT_STATUS",
    "permit expired": "B-PERMIT_STATUS",
    "main road": "B-LOCATION",
    "market area": "B-LOCATION",
    "hostel gate": "B-LOCATION",
    "bus stop": "B-LOCATION",
    "block a": "B-LOCATION"
}

def normalize_for_ner(text):
    """Normalize abbreviations for NER matching while keeping original text for model input"""
    for k, v in ABBR_REVERSE.items():
        text = re.sub(rf"\b{k}\b", v, text)
    return text

def tag_tokens(tokens, row):
    """Enhanced tagging function that handles multi-word entities"""
    tags = ["O"] * len(tokens)
    
    # Join tokens to check for multi-word entities (with normalization for matching)
    note_text = " ".join(tokens).lower()
    normalized_text = normalize_for_ner(note_text)
    
    # Check for multi-word entities first
    for entity, tag in MULTI_WORD_ENTITIES.items():
        if entity in normalized_text:
            # Find the position of this entity in tokens
            entity_tokens = entity.split()
            for i in range(len(tokens) - len(entity_tokens) + 1):
                # Use normalized text for matching but preserve original tokens
                normalized_tokens = [normalize_for_ner(t.lower()) for t in tokens]
                if " ".join(normalized_tokens[i:i+len(entity_tokens)]) == entity:
                    tags[i] = tag
                    # Tag remaining tokens as continuation of entity
                    for j in range(1, len(entity_tokens)):
                        tags[i+j] = tag.replace("B-", "I-")
    
    # Tag single tokens (only for entities not covered by multi-word matching)
    for i, token in enumerate(tokens):
        if tags[i] != "O":  # Skip if already tagged as part of multi-word entity
            continue
            
        token_lower = token.lower().strip(",.")
        normalized_token = normalize_for_ner(token_lower)
        
        if normalized_token in VEHICLES:
            tags[i] = "B-VEHICLE_TYPE"
        elif normalized_token in COLORS:
            tags[i] = "B-VEHICLE_COLOR"
        elif normalized_token == "repeat":
            tags[i] = "B-REPEAT_STATUS"
        elif normalized_token == "permit":
            tags[i] = "B-PERMIT_STATUS"
        # Handle single-word zone mentions
        elif normalized_token in ZONE_TYPES:
            # Check if next token is "zone"
            if i < len(tokens) - 1 and normalize_for_ner(tokens[i+1].lower()).startswith("zone"):
                tags[i] = "B-ZONE_TYPE"
    
    return list(zip(tokens, tags))

ner_rows = []

for _, row in df.iterrows():
    # Tokenize more carefully to preserve punctuation
    tokens = re.findall(r'\b\w+\b|[,]', row["note"])
    tagged_tokens = tag_tokens(tokens, row)
    
    for token, tag in tagged_tokens:
        ner_rows.append([token, tag])
    ner_rows.append(["", ""])  # sentence break

ner_df = pd.DataFrame(ner_rows, columns=["token", "tag"])
ner_df.to_csv("mpvpn_ner_dataset.conll", index=False, header=False)

print("Enhanced NER dataset created with multi-word entity support")
print("Features:")
print("- Abbreviation normalization for NER matching")
print("- Location entity recognition")
print("- Improved multi-word entity detection")