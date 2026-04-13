import pandas as pd
import re

df = pd.read_csv("mpvpn_comprehensive_dataset.csv")

# =============== ENTITY VOCABULARY ===============

# Vehicle types (all categories)
VEHICLES = {
    "two_wheeler": ["motorcycle", "scooter", "moped", "bike", "electric bike", "vespa"],
    "three_wheeler": ["auto rickshaw", "tuk-tuk", "auto", "e-rickshaw"],
    "four_wheeler": ["car", "sedan", "suv", "hatchback", "van", "minivan", "crossover"],
    "commercial": ["truck", "lorry", "delivery van", "pickup truck", "tractor", "bus", "minibus", "taxi", "cab"],
    "luxury": ["luxury car", "sports car", "limousine", "premium suv"],
    "specialized": ["ambulance", "police car", "fire truck", "tow truck", "garbage truck", "construction vehicle"]
}
ALL_VEHICLES = [v for vehicles in VEHICLES.values() for v in vehicles]

# Colors
COLORS = ["white", "black", "red", "blue", "silver", "grey", "green", "yellow", "brown", 
          "orange", "maroon", "navy blue", "dark grey", "light grey", "beige", "cream", "gold"]

# Vehicle brands
BRANDS = ["honda", "toyota", "maruti", "hyundai", "tata", "mahindra", "ford", "bmw", "mercedes", "audi"]

# Zone types
ZONES = [
    "residential", "commercial", "hospital", "school", "university", "college",
    "vip", "industrial", "heritage", "diplomatic", "military", "restricted",
    "residential area", "commercial district", "hospital zone", "school zone",
    "apartment complex", "housing society", "gated community", "business area",
    "shopping center", "office complex", "university campus", "government building",
    "park area", "stadium", "warehouse district", "factory zone"
]

# Locations
LOCATIONS = [
    "main road", "side street", "service road", "highway exit", "overpass", "underpass",
    "flyover", "bridge", "intersection", "market area", "shopping complex", "mall entrance",
    "store front", "restaurant area", "apartment gate", "hostel gate", "building entrance",
    "driveway", "residential street", "bus stop", "metro entrance", "railway crossing",
    "public parking", "park entrance", "fire hydrant area", "transformer box",
    "zebra crossing", "pedestrian zone", "bicycle lane", "loading zone", "disabled parking"
]

# Time-related entities
TIME_PERIODS = ["morning", "afternoon", "evening", "night", "late night", "early morning",
                "rush hour", "peak time", "lunch hour", "school hours"]

# Violation objects
VIOLATION_OBJECTS = [
    "fire hydrant", "emergency exit", "pedestrian path", "driveway", "intersection",
    "access road", "traffic lane", "turning point", "narrow road", "loading zone",
    "bus stop", "metro entrance", "railway crossing", "zebra crossing", "crosswalk",
    "sidewalk", "footpath", "ambulance route", "blind spot", "parking meter"
]

# Permit/status terms
PERMIT_TERMS = ["permit", "pass", "expired", "valid", "invalid", "unauthorized", "forged"]
REPEAT_TERMS = ["repeat", "multiple", "habitual", "known offender", "frequent violator"]

# Weather conditions
WEATHER_TERMS = ["clear", "rain", "heavy rain", "fog", "mist", "drizzle", "storm", "hot", "cold", "windy"]

# =============== ABBREVIATION NORMALIZATION ===============

ABBR_REVERSE = {
    # Common abbreviations
    "parkd": "parked", "prked": "parked", "pk": "parked",
    "veh": "vehicle", "vhcl": "vehicle",
    "blk": "blocking", "blkg": "blocking", "blckng": "blocking",
    "nr": "near", "ner": "near",
    "st": "street", "str": "street",
    "rd": "road",
    "bldg": "building", "bld": "building",
    "apt": "apartment",
    "emerg": "emergency", "emrg": "emergency",
    "viol": "violation", "violn": "violation",
    "obs": "observed", "obsvd": "observed",
    "drng": "during", "durng": "during",
    "resi": "residential", "res": "residential",
    "comm": "commercial", "coml": "commercial",
    "hosp": "hospital", "hsptl": "hospital",
    "chkd": "checked", "chked": "checked",
    "verif": "verified", "vrfd": "verified",
    "cnfrm": "confirmed", "conf": "confirmed",
    "loc": "located", "loctd": "located",
    "mrng": "morning", "morn": "morning",
    "evng": "evening", "eve": "evening",
    "nght": "night", "nt": "night",
    "hrs": "hours", "hr": "hours"
}

# =============== MULTI-WORD ENTITY RECOGNITION ===============

MULTI_WORD_ENTITIES = {
    # Violation objects
    "fire hydrant": "VIOLATION_OBJECT",
    "emergency exit": "VIOLATION_OBJECT",
    "pedestrian path": "VIOLATION_OBJECT",
    "double parking": "VIOLATION_OBJECT",
    "wrong side parking": "VIOLATION_OBJECT",
    "street cleaning": "VIOLATION_OBJECT",
    "no parking anytime": "VIOLATION_OBJECT",
    "access road": "VIOLATION_OBJECT",
    "traffic lane": "VIOLATION_OBJECT",
    "turning point": "VIOLATION_OBJECT",
    "narrow road": "VIOLATION_OBJECT",
    "loading zone": "VIOLATION_OBJECT",
    "bus stop": "VIOLATION_OBJECT",
    "metro entrance": "VIOLATION_OBJECT",
    "railway crossing": "VIOLATION_OBJECT",
    "zebra crossing": "VIOLATION_OBJECT",
    "pedestrian zone": "VIOLATION_OBJECT",
    "bicycle lane": "VIOLATION_OBJECT",
    "disabled parking": "VIOLATION_OBJECT",
    "ambulance route": "VIOLATION_OBJECT",
    "blind spot": "VIOLATION_OBJECT",
    "parking meter": "VIOLATION_OBJECT",
    
    # Zones
    "hospital zone": "ZONE_TYPE",
    "school zone": "ZONE_TYPE",
    "residential zone": "ZONE_TYPE",
    "commercial zone": "ZONE_TYPE",
    "vip zone": "ZONE_TYPE",
    "residential area": "ZONE_TYPE",
    "commercial district": "ZONE_TYPE",
    "apartment complex": "ZONE_TYPE",
    "housing society": "ZONE_TYPE",
    "gated community": "ZONE_TYPE",
    "business area": "ZONE_TYPE",
    "shopping center": "ZONE_TYPE",
    "office complex": "ZONE_TYPE",
    "university campus": "ZONE_TYPE",
    "government building": "ZONE_TYPE",
    "park area": "ZONE_TYPE",
    "warehouse district": "ZONE_TYPE",
    "factory zone": "ZONE_TYPE",
    "heritage zone": "ZONE_TYPE",
    
    # Locations
    "main road": "LOCATION",
    "side street": "LOCATION",
    "service road": "LOCATION",
    "highway exit": "LOCATION",
    "market area": "LOCATION",
    "shopping complex": "LOCATION",
    "mall entrance": "LOCATION",
    "apartment gate": "LOCATION",
    "hostel gate": "LOCATION",
    "building entrance": "LOCATION",
    "residential street": "LOCATION",
    "public parking": "LOCATION",
    "park entrance": "LOCATION",
    
    # Time
    "morning patrol": "TIME_PERIOD",
    "evening patrol": "TIME_PERIOD",
    "night patrol": "TIME_PERIOD",
    "rush hour": "TIME_PERIOD",
    "peak time": "TIME_PERIOD",
    "lunch hour": "TIME_PERIOD",
    "school hours": "TIME_PERIOD",
    "late night": "TIME_PERIOD",
    "early morning": "TIME_PERIOD",
    
    # Status
    "repeat violation": "REPEAT_STATUS",
    "multiple violations": "REPEAT_STATUS",
    "habitual offender": "REPEAT_STATUS",
    "known offender": "REPEAT_STATUS",
    "frequent violator": "REPEAT_STATUS",
    "expired permit": "PERMIT_STATUS",
    "invalid permit": "PERMIT_STATUS",
    "wrong zone": "PERMIT_STATUS",
    
    # Vehicle types (multi-word)
    "auto rickshaw": "VEHICLE_TYPE",
    "electric bike": "VEHICLE_TYPE",
    "delivery van": "VEHICLE_TYPE",
    "pickup truck": "VEHICLE_TYPE",
    "luxury car": "VEHICLE_TYPE",
    "sports car": "VEHICLE_TYPE",
    "premium suv": "VEHICLE_TYPE",
    "police car": "VEHICLE_TYPE",
    "fire truck": "VEHICLE_TYPE",
    "tow truck": "VEHICLE_TYPE",
    "garbage truck": "VEHICLE_TYPE",
    "construction vehicle": "VEHICLE_TYPE",
    
    # Colors (multi-word)
    "navy blue": "VEHICLE_COLOR",
    "dark grey": "VEHICLE_COLOR",
    "light grey": "VEHICLE_COLOR"
}

# =============== HELPER FUNCTIONS ===============

def normalize_for_ner(text):
    """Normalize abbreviations for better NER matching"""
    text_lower = text.lower()
    for abbr, full in ABBR_REVERSE.items():
        text_lower = re.sub(rf"\b{abbr}\b", full, text_lower)
    return text_lower

def extract_license_plate(text):
    """Extract license plate if present"""
    pattern = r'\b([A-Z]{2}-\d{2}-[A-Z]\d{4})\b'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).upper() if match else None

# =============== TAGGING FUNCTION ===============

def tag_tokens_comprehensive(tokens, row):
    """
    Comprehensive NER tagging with multi-word entity support
    """
    tags = ["O"] * len(tokens)
    
    # Join tokens to check for multi-word entities
    note_text = " ".join(tokens).lower()
    normalized_text = normalize_for_ner(note_text)
    
    # Track which positions are already tagged
    tagged_positions = set()
    
    # === STEP 1: Tag multi-word entities first ===
    for entity, tag_type in MULTI_WORD_ENTITIES.items():
        entity_tokens = entity.split()
        entity_len = len(entity_tokens)
        
        for i in range(len(tokens) - entity_len + 1):
            if i in tagged_positions:
                continue
            
            # Check if this sequence matches the entity
            normalized_sequence = " ".join([normalize_for_ner(t.lower()) for t in tokens[i:i+entity_len]])
            
            if normalized_sequence == entity:
                # Tag this multi-word entity
                tags[i] = f"B-{tag_type}"
                for j in range(1, entity_len):
                    tags[i+j] = f"I-{tag_type}"
                    tagged_positions.add(i+j)
                tagged_positions.add(i)
    
    # === STEP 2: Tag single-token entities ===
    for i, token in enumerate(tokens):
        if i in tagged_positions:
            continue
        
        token_clean = token.lower().strip(",.()")
        normalized_token = normalize_for_ner(token_clean)
        
        # Vehicle types
        if normalized_token in ALL_VEHICLES:
            tags[i] = "B-VEHICLE_TYPE"
            tagged_positions.add(i)
        
        # Vehicle brands
        elif normalized_token in BRANDS:
            tags[i] = "B-VEHICLE_BRAND"
            tagged_positions.add(i)
        
        # Colors (single word)
        elif normalized_token in [c.lower() for c in COLORS if ' ' not in c]:
            tags[i] = "B-VEHICLE_COLOR"
            tagged_positions.add(i)
        
        # License plate detection
        elif re.match(r'^[A-Z]{2}-\d{2}-[A-Z]\d{4}$', token.upper()):
            tags[i] = "B-LICENSE_PLATE"
            tagged_positions.add(i)
        
        # Time periods (single word)
        elif normalized_token in ["morning", "afternoon", "evening", "night"]:
            tags[i] = "B-TIME_PERIOD"
            tagged_positions.add(i)
        
        # Permit status
        elif normalized_token in ["permit", "pass"]:
            tags[i] = "B-PERMIT_STATUS"
            tagged_positions.add(i)
        elif normalized_token in ["expired", "invalid", "valid", "forged"]:
            # Check context - only tag if near "permit"
            context = " ".join([tokens[j].lower() for j in range(max(0, i-2), min(len(tokens), i+3))])
            if "permit" in context or "pass" in context:
                tags[i] = "B-PERMIT_STATUS"
                tagged_positions.add(i)
        
        # Repeat status
        elif normalized_token in ["repeat", "multiple", "habitual", "frequent"]:
            tags[i] = "B-REPEAT_STATUS"
            tagged_positions.add(i)
        
        # Weather
        elif normalized_token in WEATHER_TERMS:
            tags[i] = "B-WEATHER"
            tagged_positions.add(i)
        
        # Actions/violations
        elif normalized_token in ["blocking", "obstructing", "parked", "parking", "violating"]:
            tags[i] = "B-ACTION"
            tagged_positions.add(i)
    
    return list(zip(tokens, tags))

# =============== DATASET GENERATION ===============

def generate_ner_dataset():
    """Generate NER dataset in CoNLL format"""
    
    print("Generating comprehensive NER dataset...")
    print(f"Processing {len(df)} records...")
    
    ner_rows = []
    entity_stats = {
        "VEHICLE_TYPE": 0,
        "VEHICLE_BRAND": 0,
        "VEHICLE_COLOR": 0,
        "LICENSE_PLATE": 0,
        "ZONE_TYPE": 0,
        "LOCATION": 0,
        "TIME_PERIOD": 0,
        "VIOLATION_OBJECT": 0,
        "PERMIT_STATUS": 0,
        "REPEAT_STATUS": 0,
        "WEATHER": 0,
        "ACTION": 0
    }
    
    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"Progress: {idx}/{len(df)}")
        
        note = row["note"]
        
        # Tokenize carefully to preserve punctuation and structure
        # Split on whitespace but keep punctuation separate
        tokens = re.findall(r'\b[\w-]+\b|[(),.]', note)
        
        # Tag tokens
        tagged_tokens = tag_tokens_comprehensive(tokens, row)
        
        # Add to dataset
        for token, tag in tagged_tokens:
            ner_rows.append([token, tag])
            
            # Count entities
            if tag.startswith("B-"):
                entity_type = tag[2:]
                if entity_type in entity_stats:
                    entity_stats[entity_type] += 1
        
        # Add sentence separator
        ner_rows.append(["", ""])
    
    return ner_rows, entity_stats

# =============== MAIN EXECUTION ===============

if __name__ == "__main__":
    ner_rows, entity_stats = generate_ner_dataset()
    
    # Create DataFrame
    ner_df = pd.DataFrame(ner_rows, columns=["token", "tag"])
    
    # Save in CoNLL format (no header)
    ner_df.to_csv("mpvpn_ner_dataset.conll", index=False, header=False, sep="\t")
    
    print("\n" + "="*60)
    print("NER DATASET CREATED")
    print("="*60)
    
    print(f"\n✓ File: mpvpn_ner_dataset.conll")
    print(f"✓ Total tokens: {len(ner_df)}")
    print(f"✓ Total sentences: {ner_df[ner_df['token'] == ''].shape[0]}")
    
    print("\n" + "="*60)
    print("ENTITY STATISTICS")
    print("="*60)
    
    total_entities = sum(entity_stats.values())
    print(f"\nTotal entities tagged: {total_entities}\n")
    
    for entity_type, count in sorted(entity_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_entities * 100) if total_entities > 0 else 0
        print(f"  {entity_type:.<25} {count:>6} ({percentage:>5.1f}%)")
    
    print("\n" + "="*60)
    print("SUPPORTED ENTITY TYPES")
    print("="*60)
    
    entity_descriptions = {
        "VEHICLE_TYPE": "Type of vehicle (car, bike, truck, etc.)",
        "VEHICLE_BRAND": "Vehicle manufacturer (Honda, Toyota, etc.)",
        "VEHICLE_COLOR": "Color of vehicle",
        "LICENSE_PLATE": "Vehicle registration number",
        "ZONE_TYPE": "Zone classification (hospital zone, residential area, etc.)",
        "LOCATION": "Specific location (main road, bus stop, etc.)",
        "TIME_PERIOD": "Time of day or duration (morning, rush hour, etc.)",
        "VIOLATION_OBJECT": "Object being obstructed (fire hydrant, crosswalk, etc.)",
        "PERMIT_STATUS": "Parking permit status (expired, invalid, etc.)",
        "REPEAT_STATUS": "Repeat violation indicators",
        "WEATHER": "Weather conditions",
        "ACTION": "Violation action (blocking, parked, etc.)"
    }
    
    for entity_type, description in entity_descriptions.items():
        print(f"\n  {entity_type}")
        print(f"    {description}")
    
    print("\n" + "="*60)
    print("SAMPLE TAGGED SENTENCES")
    print("="*60)
    
    # Display first 3 complete sentences
    sentences = []
    current_sentence = []
    
    for _, row in ner_df.iterrows():
        if row['token'] == '':
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = []
            if len(sentences) >= 3:
                break
        else:
            current_sentence.append((row['token'], row['tag']))
    
    for i, sentence in enumerate(sentences, 1):
        print(f"\n{i}. Sentence:")
        tokens = [t for t, _ in sentence]
        print(f"   {' '.join(tokens)}")
        print(f"\n   Tagged entities:")
        for token, tag in sentence:
            if tag != 'O':
                print(f"     {token:.<20} {tag}")
    
    print("\n" + "="*60)
    print("FEATURES")
    print("="*60)
    print("  ✓ Multi-word entity recognition")
    print("  ✓ Abbreviation normalization")
    print("  ✓ Context-aware tagging")
    print("  ✓ License plate extraction")
    print("  ✓ 12 comprehensive entity types")
    print("  ✓ CoNLL format compatible with spaCy, transformers, etc.")
    
    print("\n✅ NER dataset generation complete!")