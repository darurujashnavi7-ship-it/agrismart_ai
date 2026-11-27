import json
import re
from pathlib import Path


GUIDANCE_PATH=Path(__file__).resolve().parents[1]/"data"/"soil_image_guidance.json"


def load_soil_guidance():
    try:
        with GUIDANCE_PATH.open(encoding="utf-8") as guidance_file:
            config=json.load(guidance_file)
    except (OSError,json.JSONDecodeError) as error:
        raise ValueError(f"Could not load soil image guidance: {error}") from error
    classes=config.get("classes") if isinstance(config,dict) else None
    source=config.get("source") if isinstance(config,dict) else None
    if not isinstance(classes,dict) or not isinstance(source,dict):
        raise ValueError("Soil image guidance must define class notes and a dataset source.")
    for soil_class,profile in classes.items():
        if not isinstance(profile,dict) or not isinstance(profile.get("description"),str):
            raise ValueError(f"Soil image guidance is invalid for {soil_class}.")
        for field in ("characteristics","advantages","limitations","improvements"):
            if not isinstance(profile.get(field),list):
                raise ValueError(f"Soil image guidance for {soil_class} needs a {field} list.")
        crop_guidance=profile.get("crop_guidance")
        if (
            not isinstance(crop_guidance,dict)
            or not isinstance(crop_guidance.get("crops"),list)
            or any(not isinstance(crop,str) or not crop.strip() for crop in crop_guidance["crops"])
            or not isinstance(crop_guidance.get("note"),str)
            or not crop_guidance["note"].strip()
            or not isinstance(crop_guidance.get("source_title"),str)
            or not crop_guidance["source_title"].strip()
            or not isinstance(crop_guidance.get("source_url"),str)
            or not crop_guidance["source_url"].startswith("https://")
        ):
            raise ValueError(f"Soil image crop guidance is invalid for {soil_class}.")
    return config


def soil_image_assistant_context(
    prediction,
    guidance,
    crop_result,
    location="",
    season="",
    irrigation="",
    climate_source="",
    weather_summary=None,
):
    predicted=prediction.get("predicted_soil")
    confidence=prediction.get("confidence")
    context=[
        "Image-based soil classification only. It is not a laboratory soil test.",
        "The image cannot measure texture, NPK, pH, organic carbon, electrical conductivity, "
        "moisture, salinity, or fertilizer requirements.",
        "Do not invent missing measurements or claim a guaranteed crop recommendation.",
    ]
    if location:
        context.append(f"Farmer-provided location: {location}.")
    if season:
        context.append(f"Farmer-selected/derived growing season: {season}.")
    if irrigation:
        context.append(f"Farmer-reported irrigation availability: {irrigation}.")
    if climate_source:
        context.append(f"Farmer-provided seasonal climate source: {climate_source}.")
    if weather_summary:
        context.append(
            "Short-range forecast context only, not seasonal climate normals: "+
            weather_summary.get("summary","")
        )
    if prediction.get("uncertain") or not predicted:
        context.append(
            "The classifier marked this image uncertain. Do not state a definite soil class; "
            "ask the farmer to retake a clear soil-surface photograph."
        )
    else:
        context.append(f"Predicted dataset class: {predicted}.")
        if isinstance(confidence,(int,float)):
            context.append(f"Model score: {confidence*100:.1f}%; not a calibrated probability.")
        profile=(
            guidance.get("classes",{}).get(predicted)
            if "classes" in guidance else guidance
        )
        if profile:
            context.append(
                "General class notes, not sample measurements: "+profile["description"]
            )
            crop_guidance=profile.get("crop_guidance",{})
            context.append(
                "Source-backed general crop examples for this class, not locally validated "
                "recommendations: "+
                (", ".join(crop_guidance.get("crops",[])) or "none specified")+"."
            )
            context.append("Crop guidance limitation: "+crop_guidance.get("note",""))
            if crop_guidance.get("source_url"):
                context.append(
                    "Crop guidance source: "+crop_guidance.get("source_title","")+
                    " "+crop_guidance["source_url"]
                )
    recommendations=crop_result.get("recommendations",[])
    if recommendations:
        context.append("Configured rule-based crop profiles:")
        for item in recommendations:
            context.append(
                f"{item['crop']} ({item['score']}/100): {item['why']} "
                "Scores are rule-based screens, not probabilities or yield predictions."
            )
    else:
        context.append(
            "No sourced local crop profiles matched or were configured. Do not give a top-five "
            "list; ask for district/state, season, water access, and locally sourced climate data."
        )
    return "\n".join(context)


def soil_image_question_is_relevant(question):
    lowered=question.casefold()
    return bool(re.search(
        r"\b(?:soil|image|photo|photograph|picture|classification|classify|"
        r"npk|ph|organic\s+carbon)\b|"
        r"\b(?:suitable|recommended|recommendation|recommend)\s+crops?\b|"
        r"\bcrops?\s+(?:suitable|recommended|recommendation)\b",
        lowered,
    ))


def soil_image_assistant_fallback(question,prediction,guidance,crop_result):
    if not soil_image_question_is_relevant(question):
        return None
    answer=[]
    if prediction.get("uncertain") or not prediction.get("predicted_soil"):
        candidates=", ".join(
            f"{item['soil_type']} ({item['confidence']*100:.1f}%)"
            for item in prediction.get("top_predictions",[])
        )
        answer.append(
            "The soil-photo classifier is uncertain and does not support a definite soil label."
        )
        if candidates:
            answer.append("Tentative model candidates: "+candidates+".")
    else:
        answer.append(
            f"The image model predicted {prediction['predicted_soil']} with a "
            f"{prediction['confidence']*100:.1f}% model score."
        )
        profile=(
            guidance.get("classes",{}).get(prediction["predicted_soil"])
            if "classes" in guidance else guidance
        )
        if profile:
            answer.append(profile["description"])
    answer.append(
        "This image score is not a calibrated probability or laboratory soil test. "
        "A photo cannot measure NPK, pH, organic carbon, electrical conductivity, "
        "salinity, or soil moisture."
    )
    recommendations=crop_result.get("recommendations",[])
    if not prediction.get("uncertain") and prediction.get("predicted_soil"):
        profile=guidance.get("classes",{}).get(prediction["predicted_soil"],guidance)
        crop_guidance=profile.get("crop_guidance",{})
        crops=crop_guidance.get("crops",[])
        if crops:
            answer.append(
                "General crop examples for this soil class: "+", ".join(crops)+"."
            )
        answer.append(crop_guidance.get(
            "note",
            "Confirm crop suitability with local agricultural guidance.",
        ))
        if crop_guidance.get("source_url"):
            answer.append(
                "Source: "+crop_guidance.get("source_title","")+
                " "+crop_guidance["source_url"]+"."
            )
    if recommendations:
        answer.append(
            "Configured rule-based crop profiles: "+
            "; ".join(
                f"{item['crop']} ({item['score']}/100): {item['why']}"
                for item in recommendations
            )+
            ". Scores are not yield predictions or guarantees."
        )
    answer.append(
        "Use a representative laboratory soil test before deciding nutrient amendments "
        "or fertilizer rates."
    )
    return " ".join(answer)
