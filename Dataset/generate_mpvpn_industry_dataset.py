import random
import pandas as pd

# ------------------- Vocab Pools -------------------
VEHICLES = ["car", "bike", "suv", "truck", "auto"]
COLORS = ["white", "black", "red", "blue", "silver"]
ZONES = ["residential", "commercial", "hospital", "school", "vip"]
LOCATIONS = ["main road", "market area", "hostel gate", "bus stop", "block a"]
WEATHER = ["clear", "rain", "fog"]
SHIFTS = ["morning", "evening", "night"]
PERMIT = ["valid", "expired", "na"]
REPEAT = [True, False]

# Updated abbreviations - more controlled and consistent
ABBR = {
    "parked": "parkd",
    "near": "nr",
    "blocking": "blk",
    "hydrant": "hydrnt",
    "vehicle": "veh",
    "entry": "entrnce",
    "street": "st",
    "cleaning": "clng"
}

ONTOLOGY = {
    "OBSTRUCTION": ["Fire Hydrant", "Emergency Exit", "Pedestrian Path"],
    "TRAFFIC_FLOW": ["Double Parking", "Wrong Side Parking"],
    "NO_PARKING": ["Street Cleaning", "No Parking Anytime"],
    "PERMIT_VIOLATION": ["Expired Permit"],
    "RESTRICTED_AREA": ["Hospital Zone", "School Zone"]
}

# Subtype-specific phrasing for more natural text
SUBTYPE_PHRASES = {
    "Fire Hydrant": ["near fire hydrant", "blocking fire hydrant"],
    "Emergency Exit": ["blocking emergency exit"],
    "Pedestrian Path": ["obstructing pedestrian path"],
    "Double Parking": ["double parked"],
    "Wrong Side Parking": ["parked on wrong side"],
    "Street Cleaning": ["during street cleaning"],
    "No Parking Anytime": ["in no parking area"],
    "Hospital Zone": ["in hospital zone"],
    "School Zone": ["in school zone"],
    "Expired Permit": ["with expired permit"]
}

# Category weights to address class imbalance
CATEGORY_WEIGHTS = {
    "OBSTRUCTION": 0.25,
    "TRAFFIC_FLOW": 0.20,
    "NO_PARKING": 0.25,
    "PERMIT_VIOLATION": 0.10,
    "RESTRICTED_AREA": 0.20
}

# ------------------- Severity Logic -------------------
def assign_severity(cat, subtype, zone, repeat):
    if subtype == "Fire Hydrant" and zone == "hospital" and repeat:
        return "Critical"
    if subtype in ["Emergency Exit", "Pedestrian Path"]:
        return "High"
    if cat == "TRAFFIC_FLOW":
        return "Medium"
    return "Low"

# ------------------- Weighted Random Choice -------------------
def weighted_choice(choices, weights):
    """Select randomly based on weights"""
    total = sum(weights)
    r = random.uniform(0, total)
    upto = 0
    for choice, weight in zip(choices, weights):
        if upto + weight >= r:
            return choice
        upto += weight
    assert False, "Shouldn't get here"

# ------------------- Note Generator -------------------
def generate_note(cat, subtype, zone, shift, permit, repeat, noise):
    v = random.choice(VEHICLES)
    c = random.choice(COLORS)
    loc = random.choice(LOCATIONS)

    # Use subtype-specific phrasing when available
    if subtype in SUBTYPE_PHRASES:
        violation_phrase = random.choice(SUBTYPE_PHRASES[subtype])
    else:
        violation_phrase = subtype.lower()

    # Create more natural sentence structures
    note_templates = [
        f"{c} {v} {violation_phrase} at {loc} during {shift} patrol",
        f"{c} {v} spotted {violation_phrase} at {loc} on {shift} shift",
        f"Found {c} {v} violating {subtype.lower()} rules at {loc} during {shift}",
        f"{shift.capitalize()} patrol: {c} {v} observed {violation_phrase} at {loc}"
    ]
    
    note = random.choice(note_templates)

    # Add zone mention sometimes for better NER signal
    if random.random() < 0.6:
        note += f" in {zone} zone"

    if repeat:
        note += ", repeat violation"
    if permit == "expired":
        note += ", permit expired"

    # Apply abbreviations more selectively to maintain data quality
    if noise == "High":
        # Only apply a subset of abbreviations to avoid over-abbreviating
        selected_abbrs = dict(random.sample(list(ABBR.items()), k=min(3, len(ABBR))))
        for k, v in selected_abbrs.items():
            note = note.replace(k, v)
    elif noise == "Medium":
        # Apply fewer abbreviations for medium noise level
        selected_abbrs = dict(random.sample(list(ABBR.items()), k=min(2, len(ABBR))))
        for k, v in selected_abbrs.items():
            note = note.replace(k, v)

    return note.lower()

# ------------------- Dataset Generation -------------------
records = []

# Get categories and weights for balanced sampling
categories = list(ONTOLOGY.keys())
weights = [CATEGORY_WEIGHTS.get(cat, 1.0/len(categories)) for cat in categories]

for _ in range(50000):
    # Use weighted sampling to address class imbalance
    cat = weighted_choice(categories, weights)
    subtype = random.choice(ONTOLOGY[cat])
    zone = random.choice(ZONES)
    shift = random.choice(SHIFTS)
    permit = random.choice(PERMIT)
    repeat = random.choice(REPEAT)
    noise = random.choice(["Low", "Medium", "High"])

    severity = assign_severity(cat, subtype, zone, repeat)
    note = generate_note(cat, subtype, zone, shift, permit, repeat, noise)

    records.append([
        note, cat, subtype, severity,
        zone, shift, repeat, permit,
        random.choice(WEATHER),
        round(random.uniform(0.6, 1.0), 2),
        noise
    ])

df = pd.DataFrame(records, columns=[
    "note", "violation_category", "violation_subtype", "severity",
    "zone_type", "patrol_shift", "repeat_flag", "permit_status",
    "weather", "confidence_level", "noise_level"
])

df.to_csv("mpvpn_industry_patrol_notes.csv", index=False)
print("Enhanced industry-level dataset created:", df.shape)
print("Sample notes:")
for i in range(3):
    print(f"  {i+1}. {df.iloc[i]['note']}")