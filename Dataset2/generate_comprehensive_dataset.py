import random
import pandas as pd
from datetime import datetime, timedelta

# =============== EXPANDED VOCABULARY POOLS ===============

# Vehicle types - comprehensive list
VEHICLES = {
    "two_wheeler": ["motorcycle", "scooter", "moped", "bike", "electric bike", "vespa"],
    "three_wheeler": ["auto rickshaw", "tuk-tuk", "auto", "e-rickshaw"],
    "four_wheeler": ["car", "sedan", "suv", "hatchback", "van", "minivan", "crossover"],
    "commercial": ["truck", "lorry", "delivery van", "pickup truck", "tractor", "bus", "minibus", "taxi", "cab"],
    "luxury": ["luxury car", "sports car", "limousine", "premium suv"],
    "specialized": ["ambulance", "police car", "fire truck", "tow truck", "garbage truck", "construction vehicle"]
}

# Vehicle makes and models for realism
VEHICLE_BRANDS = ["Honda", "Toyota", "Maruti", "Hyundai", "Tata", "Mahindra", "Ford", "BMW", "Mercedes", "Audi"]
LICENSE_PLATE_PATTERNS = ["DL-{}-{}", "MH-{}-{}", "KA-{}-{}", "TN-{}-{}", "UP-{}-{}"]

# Expanded colors
COLORS = ["white", "black", "red", "blue", "silver", "grey", "green", "yellow", "brown", "orange", 
          "maroon", "navy blue", "dark grey", "light grey", "beige", "cream", "gold"]

# Comprehensive zone types
ZONES = {
    "residential": ["residential area", "apartment complex", "housing society", "gated community", "colony"],
    "commercial": ["commercial district", "business area", "shopping center", "mall parking", "office complex"],
    "institutional": ["hospital zone", "school zone", "university campus", "college area", "government building"],
    "transportation": ["bus terminal", "railway station", "metro station", "airport area", "taxi stand"],
    "recreational": ["park area", "stadium", "theater district", "museum area", "tourist spot"],
    "industrial": ["industrial area", "warehouse district", "factory zone", "logistics hub"],
    "special": ["vip zone", "diplomatic area", "military zone", "restricted area", "heritage zone"]
}

# Detailed locations with subcategories
LOCATIONS = {
    "roads": ["main road", "side street", "service road", "highway exit", "overpass", "underpass", "flyover", "bridge", "intersection"],
    "commercial": ["market area", "shopping complex", "mall entrance", "store front", "restaurant area", "food court"],
    "residential": ["apartment gate", "hostel gate", "building entrance", "driveway", "residential street", "society entrance"],
    "public": ["bus stop", "metro entrance", "railway crossing", "public parking", "park entrance", "playground"],
    "infrastructure": ["fire hydrant area", "transformer box", "electric pole", "manhole cover", "water supply point"],
    "special": ["zebra crossing", "pedestrian zone", "bicycle lane", "loading zone", "disabled parking", "vip entrance"]
}

# Time contexts
TIME_CONTEXTS = {
    "shifts": ["morning", "afternoon", "evening", "night", "late night", "early morning"],
    "specific": ["rush hour", "peak time", "off-peak", "lunch hour", "closing time", "school hours"],
    "duration": ["for 2 hours", "overnight", "for 30 mins", "all day", "temporarily", "briefly"]
}

# Weather and environmental conditions
WEATHER_CONDITIONS = ["clear", "rain", "heavy rain", "fog", "mist", "drizzle", "storm", "hot", "cold", "windy"]
VISIBILITY = ["good visibility", "poor visibility", "limited visibility", "clear view", "obstructed view"]

# Officer/patrol information
PATROL_TYPES = ["routine patrol", "complaint-based", "spot check", "enforcement drive", "special operation"]
CONFIDENCE_DESCRIPTORS = ["verified", "confirmed", "suspected", "reported", "witnessed", "captured on camera"]

# =============== COMPREHENSIVE VIOLATION ONTOLOGY ===============

ONTOLOGY = {
    "OBSTRUCTION": {
        "subtypes": {
            "Fire Hydrant Blocking": ["blocking fire hydrant", "parked near fire hydrant", "obstructing fire hydrant access"],
            "Emergency Exit Blocking": ["blocking emergency exit", "obstructing emergency door", "parked at emergency exit"],
            "Pedestrian Path Obstruction": ["blocking footpath", "obstructing sidewalk", "parked on pedestrian crossing"],
            "Driveway Blocking": ["blocking driveway", "obstructing building entrance", "parked at gate"],
            "Intersection Blocking": ["blocking intersection", "parked at junction", "obstructing traffic flow"],
            "Access Road Blocking": ["blocking access road", "obstructing service lane", "blocking delivery access"]
        },
        "severity_factors": ["emergency_service_impact", "pedestrian_safety", "accessibility"]
    },
    
    "TRAFFIC_FLOW": {
        "subtypes": {
            "Double Parking": ["double parked", "parking in second lane", "blocking parked vehicles"],
            "Wrong Side Parking": ["parked on wrong side", "facing wrong direction", "reverse parking violation"],
            "Lane Obstruction": ["blocking traffic lane", "parked in moving lane", "obstructing flow"],
            "Turning Point Violation": ["parked at turn", "blocking turning radius", "obstructing corner"],
            "Narrow Road Parking": ["parked on narrow road", "blocking narrow passage", "single lane obstruction"]
        },
        "severity_factors": ["traffic_congestion", "accident_risk", "flow_disruption"]
    },
    
    "NO_PARKING_ZONES": {
        "subtypes": {
            "Absolute No Parking": ["in no parking zone", "parked despite signage", "violating no parking sign"],
            "Time-Restricted Parking": ["parked during restricted hours", "overstaying time limit", "parking outside allowed hours"],
            "Street Cleaning Violation": ["parked during cleaning hours", "obstructing street sweeper", "blocking maintenance"],
            "Event-Based Restriction": ["parked during event", "violating temporary restriction", "ignoring barricades"],
            "Loading Zone Violation": ["parked in loading zone", "violating commercial zone", "blocking delivery area"]
        },
        "severity_factors": ["signage_visibility", "time_of_violation", "duration"]
    },
    
    "PERMIT_VIOLATIONS": {
        "subtypes": {
            "Expired Permit": ["expired permit displayed", "outdated parking pass", "invalid permit"],
            "No Permit": ["parking without permit", "no permit displayed", "unauthorized parking"],
            "Wrong Zone Permit": ["permit for different zone", "wrong area permit", "invalid zone pass"],
            "Forged Permit": ["suspected fake permit", "tampered permit", "fraudulent pass"],
            "Visitor Violation": ["visitor exceeding stay", "guest parking violation", "temporary permit misuse"]
        },
        "severity_factors": ["permit_status", "zone_restriction", "fraud_indication"]
    },
    
    "RESTRICTED_AREAS": {
        "subtypes": {
            "Hospital Zone Violation": ["in hospital no parking zone", "blocking ambulance route", "emergency zone violation"],
            "School Zone Violation": ["parked in school zone", "violating school hours parking", "blocking school entrance"],
            "Disabled Parking Violation": ["parked in disabled spot", "unauthorized handicap parking", "blocking accessible space"],
            "VIP Area Violation": ["parked in vip zone", "restricted area violation", "security zone parking"],
            "Heritage Zone Violation": ["parked in heritage area", "monument zone violation", "protected area parking"]
        },
        "severity_factors": ["special_status", "safety_impact", "legal_implications"]
    },
    
    "SAFETY_HAZARDS": {
        "subtypes": {
            "Blind Spot Parking": ["parked at blind spot", "obstructing driver visibility", "dangerous curve parking"],
            "Near Crossing": ["parked near zebra crossing", "too close to crosswalk", "pedestrian crossing violation"],
            "Bus Stop Violation": ["parked at bus stop", "blocking public transport", "bus bay obstruction"],
            "Railway Crossing": ["parked near railway crossing", "blocking level crossing", "train crossing violation"],
            "Accident Prone Area": ["parked in accident zone", "high-risk area parking", "danger zone violation"]
        },
        "severity_factors": ["accident_risk", "public_safety", "visibility_impact"]
    },
    
    "ENVIRONMENTAL": {
        "subtypes": {
            "Idling Violation": ["engine running while parked", "excessive idling", "pollution during parking"],
            "Green Zone Violation": ["parked in eco-friendly zone", "violating low emission zone", "restricted vehicle type"],
            "Park Area Violation": ["parked in park", "blocking green space", "environmental zone parking"],
            "Water Body Proximity": ["parked near water body", "lake area violation", "river bank parking"]
        },
        "severity_factors": ["environmental_impact", "regulation_type"]
    },
    
    "METER_PAYMENT": {
        "subtypes": {
            "Expired Meter": ["parking meter expired", "payment time exceeded", "unpaid extension"],
            "No Payment": ["parked without payment", "meter not activated", "free parking violation"],
            "Meter Tampering": ["tampered parking meter", "damaged meter", "meter bypass attempt"],
            "Wrong Payment Method": ["invalid payment", "payment method violation", "digital payment failure"]
        },
        "severity_factors": ["payment_status", "duration", "fraud_attempt"]
    },
    
    "REPEAT_OFFENSES": {
        "subtypes": {
            "Habitual Offender": ["repeat offender", "multiple violations", "frequent violator"],
            "Same Location Repeat": ["repeat violation same spot", "habitual parking here", "known violator location"],
            "Escalating Violations": ["increasing violation frequency", "pattern of violations", "escalating behavior"],
            "Ignored Citations": ["previous tickets unpaid", "ignored warnings", "outstanding fines"]
        },
        "severity_factors": ["violation_history", "payment_compliance", "behavioral_pattern"]
    }
}

# =============== CONTEXTUAL FACTORS ===============

AGGRAVATING_FACTORS = [
    "during emergency", "despite warning", "after towing notice", "in front of officer",
    "aggressive behavior", "refused to move", "driver absent", "vehicle locked",
    "blocking multiple vehicles", "causing traffic jam", "near accident", "complaint received"
]

MITIGATING_FACTORS = [
    "vehicle breakdown", "medical emergency", "brief stop", "driver present",
    "cooperative attitude", "first offense", "moved immediately", "minimal obstruction"
]

# =============== ABBREVIATIONS & INFORMAL LANGUAGE ===============

ABBREVIATIONS = {
    "parked": ["parkd", "prked", "pk"],
    "vehicle": ["veh", "vhcl"],
    "blocking": ["blk", "blkg", "blckng"],
    "near": ["nr", "ner"],
    "street": ["st", "str"],
    "road": ["rd"],
    "building": ["bldg", "bld"],
    "apartment": ["apt"],
    "emergency": ["emerg", "emrg"],
    "violation": ["viol", "violn"],
    "observed": ["obs", "obsvd"],
    "during": ["drng", "durng"],
    "residential": ["resi", "res"],
    "commercial": ["comm", "coml"],
    "hospital": ["hosp", "hsptl"],
    "checked": ["chkd", "chked"],
    "verified": ["verif", "vrfd"],
    "confirmed": ["cnfrm", "conf"],
    "located": ["loc", "loctd"],
    "morning": ["mrng", "morn"],
    "evening": ["evng", "eve"],
    "night": ["nght", "nt"],
    "hours": ["hrs", "hr"]
}

INFORMAL_PATTERNS = {
    "found": ["spotted", "saw", "noticed", "came across"],
    "parked": ["sitting", "standing", "left", "dumped"],
    "vehicle": ["ride", "wheels"],
    "blocking": ["in the way", "stopping", "preventing access to"],
    "violation": ["issue", "problem", "breach"]
}

# =============== HELPER FUNCTIONS ===============

def generate_license_plate():
    """Generate realistic license plate"""
    pattern = random.choice(LICENSE_PLATE_PATTERNS)
    return pattern.format(
        str(random.randint(1, 99)).zfill(2),
        random.choice(["A", "B", "C", "D"]) + str(random.randint(1000, 9999))
    )

def apply_abbreviations(text, noise_level):
    """Apply abbreviations based on noise level"""
    if noise_level == "Low":
        return text
    
    abbr_chance = 0.3 if noise_level == "Medium" else 0.6
    
    for word, abbr_list in ABBREVIATIONS.items():
        if random.random() < abbr_chance and word in text:
            text = text.replace(word, random.choice(abbr_list), 1)
    
    return text

def apply_informal_language(text, noise_level):
    """Apply informal patterns"""
    if noise_level == "Low" or random.random() > 0.3:
        return text
    
    for formal, informal_list in INFORMAL_PATTERNS.items():
        if formal in text:
            text = text.replace(formal, random.choice(informal_list), 1)
    
    return text

def calculate_severity(category, subtype, zone, repeat, aggravating, time_context, weather):
    """Enhanced severity calculation with multiple factors"""
    score = 0
    
    # Base severity by category
    base_scores = {
        "OBSTRUCTION": 7,
        "SAFETY_HAZARDS": 8,
        "RESTRICTED_AREAS": 7,
        "TRAFFIC_FLOW": 5,
        "NO_PARKING_ZONES": 4,
        "PERMIT_VIOLATIONS": 4,
        "ENVIRONMENTAL": 3,
        "METER_PAYMENT": 2,
        "REPEAT_OFFENSES": 6
    }
    score += base_scores.get(category, 5)
    
    # Critical subtypes
    critical_subtypes = [
        "Fire Hydrant Blocking", "Emergency Exit Blocking", "Hospital Zone Violation",
        "Ambulance Route", "Railway Crossing", "Blind Spot Parking"
    ]
    if subtype in critical_subtypes:
        score += 3
    
    # Zone multipliers
    critical_zones = ["hospital", "school", "vip", "emergency"]
    if any(z in zone.lower() for z in critical_zones):
        score += 2
    
    if repeat:
        score += 3
    
    if aggravating:
        score += 2
    
    if "rush" in time_context or "peak" in time_context:
        score += 1
    
    if weather in ["heavy rain", "fog", "storm"]:
        score += 1
    
    # Convert score to severity level
    if score >= 12:
        return "Critical"
    elif score >= 9:
        return "High"
    elif score >= 6:
        return "Medium"
    else:
        return "Low"

def generate_comprehensive_note(category, subtype, zone_type, zone_desc, location_type, 
                                location_desc, shift, time_context, duration, permit_status, 
                                repeat, weather, visibility, aggravating, mitigating, 
                                patrol_type, confidence, noise_level):
    """Generate realistic, comprehensive parking violation note"""
    
    # Select vehicle details
    vehicle_category = random.choice(list(VEHICLES.keys()))
    vehicle = random.choice(VEHICLES[vehicle_category])
    color = random.choice(COLORS)
    
    plate = generate_license_plate() if random.random() > 0.3 else None
    brand = random.choice(VEHICLE_BRANDS) if random.random() > 0.7 else None
    
    # Build note components
    components = []
    
    # Time and patrol context
    time_intro = random.choice([
        f"During {shift} {patrol_type}",
        f"{shift.capitalize()} shift",
        f"At {time_context}",
        f"{confidence.capitalize()}"
    ])
    components.append(time_intro)
    
    # Vehicle description
    vehicle_desc = f"{color} {vehicle}"
    if brand:
        vehicle_desc = f"{color} {brand} {vehicle}"
    if plate:
        vehicle_desc += f" ({plate})"
    
    # Violation description
    violation_phrases = ONTOLOGY[category]["subtypes"][subtype]
    violation = random.choice(violation_phrases)
    
    # Location description
    location_full = f"{location_desc}"
    if zone_desc != location_desc:
        location_full += f" in {zone_desc}"
    
    # Construct main sentence
    sentence_patterns = [
        f"observed {vehicle_desc} {violation} at {location_full}",
        f"found {vehicle_desc} {violation} near {location_full}",
        f"{vehicle_desc} spotted {violation} at {location_full}",
        f"violation detected: {vehicle_desc} {violation} at {location_full}",
        f"{vehicle_desc} illegally {violation} at {location_full}"
    ]
    
    main_sentence = random.choice(sentence_patterns)
    components.append(main_sentence)
    
    # Add contextual details
    details = []
    
    if duration and random.random() > 0.4:
        details.append(f"parked {duration}")
    
    if repeat:
        repeat_phrases = ["repeat violation", "known offender", "multiple violations", "habitual offender"]
        details.append(random.choice(repeat_phrases))
    
    if permit_status not in ["valid", "na"]:
        if permit_status == "expired":
            details.append("permit expired")
        elif permit_status == "none":
            details.append("no permit displayed")
        elif permit_status == "wrong_zone":
            details.append("invalid zone permit")
    
    if weather != "clear" and random.random() > 0.5:
        details.append(f"{weather} conditions")
    
    if visibility and "poor" in visibility and random.random() > 0.6:
        details.append(visibility)
    
    if aggravating and random.random() > 0.5:
        details.append(aggravating)
    
    if mitigating and random.random() > 0.3:
        details.append(mitigating)
    
    # Assemble full note
    note = ". ".join(components)
    if details:
        note += ". " + ", ".join(details)
    
    # Apply noise/informality
    note = apply_informal_language(note, noise_level)
    note = apply_abbreviations(note, noise_level)
    
    return note.lower()

# =============== DATASET GENERATION ===============

def generate_comprehensive_dataset(num_records=100000):
    """Generate comprehensive parking violation dataset"""
    
    records = []
    
    # Category weights for realistic distribution
    category_weights = {
        "NO_PARKING_ZONES": 0.25,
        "OBSTRUCTION": 0.20,
        "TRAFFIC_FLOW": 0.15,
        "PERMIT_VIOLATIONS": 0.15,
        "RESTRICTED_AREAS": 0.10,
        "SAFETY_HAZARDS": 0.07,
        "METER_PAYMENT": 0.05,
        "ENVIRONMENTAL": 0.02,
        "REPEAT_OFFENSES": 0.01
    }
    
    categories = list(category_weights.keys())
    weights = list(category_weights.values())
    
    print(f"Generating {num_records} comprehensive parking violation records...")
    
    for i in range(num_records):
        if i % 10000 == 0:
            print(f"Progress: {i}/{num_records}")
        
        category = random.choices(categories, weights=weights)[0]
        subtype = random.choice(list(ONTOLOGY[category]["subtypes"].keys()))
        
        zone_type = random.choice(list(ZONES.keys()))
        zone_desc = random.choice(ZONES[zone_type])
        
        location_type = random.choice(list(LOCATIONS.keys()))
        location_desc = random.choice(LOCATIONS[location_type])
        
        shift = random.choice(TIME_CONTEXTS["shifts"])
        time_context = random.choice(TIME_CONTEXTS["specific"])
        duration = random.choice(TIME_CONTEXTS["duration"]) if random.random() > 0.5 else None
        
        permit_status = random.choices(
            ["valid", "expired", "none", "wrong_zone", "na"],
            weights=[0.3, 0.15, 0.15, 0.1, 0.3]
        )[0]
        
        repeat = random.random() < 0.15
        
        weather = random.choice(WEATHER_CONDITIONS)
        visibility = random.choice(VISIBILITY) if random.random() > 0.7 else None
        
        aggravating = random.choice(AGGRAVATING_FACTORS) if random.random() > 0.8 else None
        mitigating = random.choice(MITIGATING_FACTORS) if random.random() > 0.9 else None
        
        patrol_type = random.choice(PATROL_TYPES)
        confidence = random.choice(CONFIDENCE_DESCRIPTORS)
        noise_level = random.choices(["Low", "Medium", "High"], weights=[0.4, 0.4, 0.2])[0]
        
        severity = calculate_severity(
            category, subtype, zone_desc, repeat, aggravating, time_context, weather
        )
        
        note = generate_comprehensive_note(
            category, subtype, zone_type, zone_desc, location_type, location_desc,
            shift, time_context, duration, permit_status, repeat, weather, visibility,
            aggravating, mitigating, patrol_type, confidence, noise_level
        )
        
        record = {
            "note": note,
            "violation_category": category,
            "violation_subtype": subtype,
            "severity": severity,
            "zone_type": zone_type,
            "zone_description": zone_desc,
            "location_type": location_type,
            "location_description": location_desc,
            "patrol_shift": shift,
            "time_context": time_context,
            "duration": duration,
            "repeat_flag": repeat,
            "permit_status": permit_status,
            "weather": weather,
            "visibility": visibility,
            "aggravating_factor": aggravating,
            "mitigating_factor": mitigating,
            "patrol_type": patrol_type,
            "confidence_level": confidence,
            "noise_level": noise_level
        }
        
        records.append(record)
    
    return pd.DataFrame(records)

# =============== MAIN EXECUTION ===============

if __name__ == "__main__":
    df = generate_comprehensive_dataset(num_records=100000)
    
    df.to_csv("mpvpn_comprehensive_dataset.csv", index=False)
    print(f"\n✓ Comprehensive dataset created: {df.shape}")
    
    print("\n=== DATASET STATISTICS ===")
    print(f"\nViolation Categories:")
    print(df['violation_category'].value_counts())
    print(f"\nSeverity Distribution:")
    print(df['severity'].value_counts())
    print(f"\nZone Types:")
    print(df['zone_type'].value_counts())
    print(f"\nRepeat Violations: {df['repeat_flag'].sum()} ({df['repeat_flag'].sum()/len(df)*100:.1f}%)")
    
    print("\n=== SAMPLE NOTES ===")
    for i in range(5):
        row = df.iloc[i]
        print(f"\n{i+1}. Category: {row['violation_category']}")
        print(f"   Subtype: {row['violation_subtype']}")
        print(f"   Severity: {row['severity']}")
        print(f"   Note: {row['note']}")