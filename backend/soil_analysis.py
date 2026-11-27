import json
import math
from pathlib import Path

import numpy as np


SOIL_TYPES=(
    "Red soil",
    "Black soil",
    "Alluvial soil",
    "Sandy soil",
    "Clay soil",
    "Loamy soil",
)

NUMERIC_LIMITS={
    "nitrogen":(0,10000),
    "phosphorus":(0,10000),
    "potassium":(0,10000),
    "ph":(0,14),
    "moisture":(0,100),
    "temperature":(-20,80),
    "ec":(0,100),
    "organic_carbon":(0,100),
}

NUTRIENT_NAMES={
    "nitrogen":"Nitrogen",
    "phosphorus":"Phosphorus",
    "potassium":"Potassium",
}
SOIL_CONFIG_PATH=Path(__file__).resolve().parents[1]/"data"/"soil_reference_profiles.json"


def load_soil_config():
    try:
        with SOIL_CONFIG_PATH.open(encoding="utf-8") as config_file:
            config=json.load(config_file)
    except (OSError,json.JSONDecodeError) as error:
        raise ValueError(f"Could not load soil reference configuration: {error}") from error
    reference_profiles=config.get("reference_profiles")
    crop_profiles=config.get("crop_profiles")
    if not isinstance(reference_profiles,list) or not reference_profiles:
        raise ValueError("Soil reference configuration needs at least one reference profile.")
    if not isinstance(crop_profiles,list):
        raise ValueError("Crop profiles must be a list.")
    if crop_profiles and len(crop_profiles)<5:
        raise ValueError("Configure at least five crop profiles before enabling crop ranking.")
    for profile in reference_profiles:
        if not isinstance(profile,dict):
            raise ValueError("A soil reference profile must be an object.")
        if not isinstance(profile.get("name"),str) or not isinstance(profile.get("thresholds"),dict):
            raise ValueError("A soil reference profile needs a name and thresholds.")
        validate_reference(profile["thresholds"])
    for profile in crop_profiles:
        try:
            if not isinstance(profile.get("name"),str) or not profile["name"].strip():
                raise ValueError
            lower=float(profile["ph_min"])
            upper=float(profile["ph_max"])
            if not math.isfinite(lower) or not math.isfinite(upper) or not 0<=lower<upper<=14:
                raise ValueError
            if not isinstance(profile.get("soil_types"),list):
                raise ValueError
        except (KeyError,AttributeError,TypeError,ValueError) as error:
            raise ValueError("A crop profile has invalid pH or soil-type settings.") from error
    scoring=config.get("crop_scoring",{})
    if not isinstance(scoring,dict):
        raise ValueError("Crop suitability scoring configuration must be an object.")
    weights=(
        float(scoring.get("ph_weight",0)),
        float(scoring.get("soil_type_weight",0)),
        float(scoring.get("nutrient_weight",0)),
    )
    if any(not math.isfinite(weight) or weight<0 for weight in weights) or not math.isclose(sum(weights),1):
        raise ValueError("Crop suitability scoring weights must be non-negative and add up to 1.")
    penalty=float(scoring.get("ph_distance_penalty",30))
    if not math.isfinite(penalty) or penalty<0:
        raise ValueError("Crop suitability pH penalty must be a non-negative number.")
    return config


def select_soil_reference(config,soil_type,crop,location):
    normalized_soil=soil_type.casefold()
    normalized_crop=crop.casefold()
    normalized_location=location.strip().casefold()
    candidates=[]
    for index,profile in enumerate(config["reference_profiles"]):
        matches=[]
        for key,value in (
            ("region",normalized_location),
            ("soil_type",normalized_soil),
            ("crop",normalized_crop),
        ):
            target=str(profile.get(key,"*")).strip().casefold()
            matched=(
                _region_matches(target,location)
                if key=="region" else
                target in {"","*"} or target==value
            )
            if not matched:
                break
            matches.append(target not in {"","*"})
        else:
            candidates.append((sum(matches),-index,profile))
    if not candidates:
        raise ValueError("No compatible soil reference profile is configured.")
    _,_,selected=max(candidates,key=lambda item:(item[0],item[1]))
    reference=dict(selected["thresholds"])
    reference["name"]=selected["name"]
    reference["sources"]=selected.get("sources",[])
    reference["nutrient_basis"]=selected.get("nutrient_basis","")
    reference["scope"]=selected
    reference["crop_scoring"]=config["crop_scoring"]
    validate_reference(reference)
    return reference


def validate_soil_inputs(values,soil_type,crop,location):
    cleaned={}
    errors=[]
    for field,(minimum,maximum) in NUMERIC_LIMITS.items():
        value=values.get(field)
        if value is None or value=="":
            errors.append(f"{field.replace('_',' ').title()} is required.")
            continue
        try:
            number=float(value)
        except (TypeError,ValueError):
            errors.append(f"{field.replace('_',' ').title()} must be a number.")
            continue
        if not np.isfinite(number) or not minimum<=number<=maximum:
            errors.append(
                f"{field.replace('_',' ').title()} must be between {minimum} and {maximum}."
            )
            continue
        cleaned[field]=number

    if soil_type not in SOIL_TYPES:
        errors.append("Choose a soil type.")
    if not isinstance(crop,str) or not crop.strip():
        errors.append("Choose a crop.")
    if not isinstance(location,str) or not location.strip():
        errors.append("Enter the farm location.")
    elif len(location.strip())>200:
        errors.append("Farm location must be 200 characters or fewer.")
    if errors:
        raise ValueError(" ".join(errors))
    return cleaned


def _classify_three_levels(value,thresholds):
    low_upper=thresholds.get("low_upper")
    medium_upper=thresholds.get("medium_upper")
    if low_upper is None or medium_upper is None:
        return "Not configured"
    if value<float(low_upper):
        return "Low"
    if value<=float(medium_upper):
        return "Medium"
    return "High"


def validate_reference(reference):
    try:
        for nutrient in NUTRIENT_NAMES:
            ranges=reference["nutrients"][nutrient]
            _validate_threshold_pair(ranges,"low_upper","medium_upper",0,None)
        _validate_threshold_pair(reference["ph"],"acidic_upper","alkaline_lower",0,14)
        carbon=reference["organic_carbon"]
        _validate_threshold_pair(carbon,"low_upper","medium_upper",0,None)
        moisture=reference.get("moisture",{})
        _validate_threshold_pair(moisture,"dry_upper","wet_lower",0,100)
    except (KeyError,TypeError,ValueError) as error:
        raise ValueError("Reference thresholds are invalid; check that each lower cutoff is below its upper cutoff.") from error


def _validate_threshold_pair(thresholds,lower_key,upper_key,minimum,maximum):
    lower=thresholds.get(lower_key)
    upper=thresholds.get(upper_key)
    if lower is None and upper is None:
        return
    lower=float(lower)
    upper=float(upper)
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower<minimum
        or maximum is not None and upper>maximum
        or upper<=lower
    ):
        raise ValueError


def _region_matches(profile_region,location):
    region=str(profile_region or "*").strip().casefold()
    if region in {"","*"}:
        return True
    location_parts={
        part.strip().casefold()
        for part in location.split(",")
        if part.strip()
    }
    return region==location.strip().casefold() or region in location_parts


def _crop_suitability(values,soil_type,nutrient_status,crop_profiles,location,scoring):
    if len(crop_profiles)<5:
        return []
    ranked=[]
    low_nutrients=[
        NUTRIENT_NAMES[key]
        for key,status in nutrient_status.items()
        if status=="Low"
    ]
    general_profiles={}
    regional_profiles={}
    for profile in crop_profiles:
        if _region_matches(profile.get("region","*"),location):
            region=str(profile.get("region","*")).strip()
            target=regional_profiles if region not in {"","*"} else general_profiles
            target[profile["name"]]=profile
    selected_profiles={**general_profiles,**regional_profiles}
    if len(selected_profiles)<5:
        return []

    ph_weight=float(scoring.get("ph_weight",.6))
    soil_weight=float(scoring.get("soil_type_weight",.25))
    nutrient_weight=float(scoring.get("nutrient_weight",.15))
    ph_penalty=float(scoring.get("ph_distance_penalty",30))
    for profile in selected_profiles.values():
        ph_min=float(profile["ph_min"])
        ph_max=float(profile["ph_max"])
        distance=max(ph_min-values["ph"],0,values["ph"]-ph_max)
        ph_score=max(0,100-ph_penalty*distance)
        compatible_soils=[value.casefold() for value in profile.get("soil_types",[])]
        soil_score=100 if "*" in compatible_soils or soil_type.casefold() in compatible_soils else 45
        nutrient_score=max(0,100-20*len(low_nutrients))
        score=round(
            ph_score*ph_weight+soil_score*soil_weight+nutrient_score*nutrient_weight
        )
        reasons=[]
        if distance==0:
            reasons.append(f"pH {values['ph']:g} is within the profile range ({ph_min:g}–{ph_max:g}).")
        else:
            reasons.append(f"pH {values['ph']:g} is outside the profile range ({ph_min:g}–{ph_max:g}).")
        if soil_score==100:
            reasons.append(f"{soil_type} is listed as compatible in this profile.")
        else:
            reasons.append(f"{soil_type} is not listed as a preferred soil in this profile.")
        improvements=[]
        if distance:
            improvements.append("Discuss crop-specific pH correction with a local soil-testing laboratory.")
        if low_nutrients:
            improvements.append(
                "Address low " + ", ".join(low_nutrients) +
                " using crop-specific local recommendations."
            )
        if not improvements:
            improvements.append("Confirm suitability against local climate, water availability, and extension advice.")
        ranked.append({
            "crop":profile["name"],
            "score":score,
            "reasons":reasons,
            "improvements":improvements,
        })
    return sorted(ranked,key=lambda item:(-item["score"],item["crop"]))[:5]


def analyze_soil(values,soil_type,crop,location,reference,crop_profiles):
    validate_reference(reference)
    values=validate_soil_inputs(values,soil_type,crop,location)
    nutrient_status={
        key:_classify_three_levels(values[key],reference["nutrients"][key])
        for key in NUTRIENT_NAMES
    }

    ph_reference=reference["ph"]
    if ph_reference.get("acidic_upper") is None:
        ph_status="Not configured"
    elif values["ph"]<float(ph_reference["acidic_upper"]):
        ph_status="Acidic"
    elif values["ph"]>float(ph_reference["alkaline_lower"]):
        ph_status="Alkaline"
    else:
        ph_status="Neutral"

    organic_reference=reference["organic_carbon"]
    organic_status=_classify_three_levels(values["organic_carbon"],organic_reference)

    moisture_reference=reference.get("moisture",{})
    moisture_status="Not configured"
    if moisture_reference.get("dry_upper") is not None and moisture_reference.get("wet_lower") is not None:
        if values["moisture"]<float(moisture_reference["dry_upper"]):
            moisture_status="Dry"
        elif values["moisture"]>float(moisture_reference["wet_lower"]):
            moisture_status="Wet"
        else:
            moisture_status="Adequate"

    statuses={
        **nutrient_status,
        "pH":ph_status,
        "Organic carbon":organic_status,
        "Moisture":moisture_status,
    }
    in_range=[
        status=="Medium"
        for status in nutrient_status.values()
        if status!="Not configured"
    ]
    if ph_status!="Not configured":
        in_range.append(ph_status=="Neutral")
    if organic_status!="Not configured":
        in_range.append(organic_status in {"Medium","High"})
    if moisture_status!="Not configured":
        in_range.append(moisture_status=="Adequate")
    score=round(float(np.mean(in_range))*100) if in_range else None
    health_status=(
        "Not configured" if score is None else
        "Needs attention" if score<40 else
        "Monitor and improve" if score<70 else
        "Within configured ranges"
    )

    deficiencies=[
        NUTRIENT_NAMES[key]
        for key,status in nutrient_status.items()
        if status=="Low"
    ]
    recommendations=[]
    fertilizer_options={
        "nitrogen":("Nitrogen source (such as urea)","The configured reference classifies nitrogen as low."),
        "phosphorus":("Phosphorus source (such as single superphosphate or DAP)","The configured reference classifies phosphorus as low."),
        "potassium":("Potassium source (such as MOP or SOP)","The configured reference classifies potassium as low."),
    }
    for key,(title,reason) in fertilizer_options.items():
        if nutrient_status[key]=="Low":
            recommendations.append({
                "category":"Nutrient",
                "title":title,
                "reason":reason+" Confirm product and rate using the crop, region, soil-test method, and local recommendation.",
            })
        elif nutrient_status[key]=="High":
            recommendations.append({
                "category":"Nutrient",
                "title":f"Avoid adding extra {NUTRIENT_NAMES[key].lower()} without a crop plan",
                "reason":"The configured reference classifies this nutrient as high; further application may be unnecessary.",
            })
    if organic_status=="Low":
        recommendations.append({
            "category":"Organic matter",
            "title":"Consider mature compost or well-decomposed farmyard manure",
            "reason":"The configured organic-carbon reference classifies this result as low; organic amendments can add carbon, but choose material and timing for the crop and region.",
        })
    if ph_status=="Acidic":
        recommendations.append({
            "category":"Soil pH",
            "title":"Review acidity correction with a soil-testing laboratory",
            "reason":"The entered pH is below the configured neutral range. Lime requirement depends on soil buffering and a laboratory recommendation; no rate is inferred here.",
        })
    elif ph_status=="Alkaline":
        recommendations.append({
            "category":"Soil pH",
            "title":"Confirm alkalinity and any sodicity before treatment",
            "reason":"The entered pH is above the configured neutral range. Amendments such as gypsum are appropriate only for diagnosed sodicity and a local recommendation.",
        })
    if moisture_status=="Dry":
        recommendations.append({
            "category":"Water management",
            "title":"Check root-zone moisture and irrigation need",
            "reason":"The value is below the user-configured moisture threshold; irrigation need also depends on crop stage, texture, and the sensor method.",
        })
    elif moisture_status=="Wet":
        recommendations.append({
            "category":"Water management",
            "title":"Check drainage before irrigating",
            "reason":"The value is above the user-configured moisture threshold; avoid assuming the same threshold applies to every soil or sensor.",
        })
    if not deficiencies and all(status!="Not configured" for status in nutrient_status.values()):
        deficiencies=["No low N, P, or K class under the selected reference profile."]
    elif not deficiencies:
        deficiencies=["NPK reference thresholds are not fully configured."]

    crop_suitability=_crop_suitability(
        values,
        soil_type,
        nutrient_status,
        crop_profiles,
        location,
        reference.get("crop_scoring",{}),
    )
    crop_scope=(
        "not configured" if len(crop_profiles)<5 else
        "no location-matched profiles" if not crop_suitability else
        "location-matched" if any(
            str(profile.get("region","*")).strip().casefold() not in {"","*"}
            and _region_matches(profile.get("region","*"),location)
            for profile in crop_profiles
        ) else "general soil-only"
    )
    return {
        "values":values,
        "soil_type":soil_type,
        "crop":crop.strip(),
        "location":location.strip(),
        "nutrient_status":nutrient_status,
        "ph_status":ph_status,
        "organic_carbon_status":organic_status,
        "moisture_status":moisture_status,
        "statuses":statuses,
        "health_score":score,
        "health_status":health_status,
        "deficiencies":deficiencies,
        "recommendations":recommendations,
        "crop_suitability":crop_suitability,
        "crop_suitability_scope":crop_scope,
        "reference_name":reference.get("name","User-configured reference"),
        "reference_sources":reference.get("sources",[]),
    }


def soil_assistant_fallback(question,analysis):
    lowered=question.lower()
    values=analysis["values"]
    if "moisture" in lowered or "irrigat" in lowered or "water" in lowered:
        return (
            f"The entered soil-moisture reading is {values['moisture']:g}%. "
            f"Its configured status is {analysis['moisture_status'].lower()}. "
            "Check crop stage, root-zone depth, weather, and the sensor's soil-specific calibration before deciding when to irrigate."
        )
    if "crop" in lowered:
        if not analysis["crop_suitability"]:
            if analysis["crop_suitability_scope"]!="not configured":
                return (
                    "No top-five crop suggestions are available because fewer than five verified "
                    "crop profiles match this farm location. Add locally approved regional profiles "
                    "before using crop suitability results."
                )
            return (
                "No top-five crop suggestions are available because the project has no verified "
                "crop-specific soil profiles configured. Add locally approved pH and soil-type "
                "profiles before using crop suitability results."
            )
        crops=", ".join(
            f"{item['crop']} ({item['score']}/100)"
            for item in analysis["crop_suitability"]
        )
        return (
            f"Based only on the configured soil pH and soil-type rules, the top suggestions are: {crops}. "
            "This is a rule-based soil screen, not an ML prediction; climate, water access, and local crop calendars are not included."
        )
    if "ph" in lowered or "acid" in lowered or "alkaline" in lowered:
        return (
            f"The entered pH is {values['ph']:g} and is classified as {analysis['ph_status'].lower()} "
            "under the selected configurable reference. Confirm any amendment rate with a local laboratory; soil buffering matters."
        )
    if any(term in lowered for term in ("nitrogen","phosphorus","potassium","fertiliz","nutrient","npk")):
        summary=", ".join(
            f"{NUTRIENT_NAMES[key]}: {status.lower()}"
            for key,status in analysis["nutrient_status"].items()
        )
        advice=" Low classes need crop- and region-specific advice; no fertilizer rate is inferred."
        return f"From the values you entered, the configured nutrient classes are {summary}.{advice}"
    return (
        f"Your entered values are classified as {analysis['health_status'].lower()} "
        f"under “{analysis['reference_name']}”, with a rule-based indicator of "
        f"{analysis['health_score'] if analysis['health_score'] is not None else 'not configured'}"
        f"{'/100' if analysis['health_score'] is not None else ''}. "
        "This is not an independent laboratory test or a machine-learning diagnosis. Ask about nutrients, pH, moisture, or crop suitability for details."
    )


def soil_assistant_context(analysis):
    values=analysis["values"]
    nutrients=", ".join(
        f"{NUTRIENT_NAMES[key]}={values[key]:g} kg/ha ({analysis['nutrient_status'][key]})"
        for key in NUTRIENT_NAMES
    )
    crops=", ".join(
        f"{item['crop']} ({item['score']}/100)"
        for item in analysis["crop_suitability"]
    )
    score_text=(
        f"{analysis['health_score']}/100"
        if analysis["health_score"] is not None
        else "not configured"
    )
    return (
        "Farmer-provided soil analysis context. These values were entered or extracted from a report "
        "and are not independently verified laboratory results. Do not invent missing measurements, "
        "claim a physical test was performed, or provide fertilizer application rates. "
        f"Crop={analysis['crop']}; location={analysis['location']}; soil type={analysis['soil_type']}; "
        f"{nutrients}; pH={values['ph']:g} ({analysis['ph_status']}); "
        f"organic carbon={values['organic_carbon']:g}% ({analysis['organic_carbon_status']}); "
        f"moisture={values['moisture']:g}% ({analysis['moisture_status']}); "
        f"soil temperature={values['temperature']:g} C; EC={values['ec']:g}; "
        f"configured reference={analysis['reference_name']}; "
        f"rule-based indicator={score_text} ({analysis['health_status']}); "
        f"rule-based, soil-only crop suggestions={crops}. Climate and local crop calendars were not assessed."
    )
