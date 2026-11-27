import json
import math
from pathlib import Path


PROFILE_PATH=Path(__file__).resolve().parents[1]/"data"/"soil_image_crop_profiles.json"
_SCORE_WEIGHTS={
    "soil":0.30,
    "region":0.20,
    "season":0.15,
    "temperature":0.15,
    "rainfall":0.15,
    "irrigation":0.05,
}


def load_crop_profiles():
    try:
        with PROFILE_PATH.open(encoding="utf-8") as profile_file:
            config=json.load(profile_file)
    except (OSError,json.JSONDecodeError) as error:
        raise ValueError(f"Could not load crop suitability profiles: {error}") from error
    profiles=config.get("profiles") if isinstance(config,dict) else None
    if not isinstance(profiles,list):
        raise ValueError("Crop suitability configuration must contain a profiles list.")
    for profile in profiles:
        _validate_profile(profile)
    return config


def _validate_profile(profile):
    required_strings=(
        "crop","water_requirement","why","growing_conditions","cultivation",
    )
    if not isinstance(profile,dict) or any(
        not isinstance(profile.get(field),str) or not profile[field].strip()
        for field in required_strings
    ):
        raise ValueError("Each crop profile must provide its name and farmer-facing guidance.")
    for field in ("soil_classes","regions","seasons","irrigation_options","improvements","sources"):
        if not isinstance(profile.get(field),list):
            raise ValueError(f"Crop profile {profile.get('crop','')} needs a {field} list.")
    for field in ("soil_classes","regions","seasons","irrigation_options","improvements"):
        if any(not isinstance(value,str) or not value.strip() for value in profile[field]):
            raise ValueError(f"Crop profile {profile['crop']} has invalid {field} values.")
    if not profile["soil_classes"] or not profile["regions"] or not profile["seasons"]:
        raise ValueError(f"Crop profile {profile['crop']} needs soil, region, and season coverage.")
    if not profile["sources"] or any(
        not isinstance(source,dict)
        or not isinstance(source.get("title"),str)
        or not source["title"].strip()
        or not isinstance(source.get("url"),str)
        or not source["url"].startswith(("https://","http://"))
        for source in profile["sources"]
    ):
        raise ValueError(f"Crop profile {profile['crop']} needs at least one source URL.")
    for field in ("temperature_c_range","seasonal_rainfall_mm_range"):
        limits=profile.get(field)
        if (
            not isinstance(limits,list)
            or len(limits)!=2
            or any(not isinstance(value,(int,float)) or not math.isfinite(value) for value in limits)
            or limits[0]>limits[1]
        ):
            raise ValueError(f"Crop profile {profile['crop']} has an invalid {field}.")


def _range_score(value,limits):
    lower,upper=limits
    if lower<=value<=upper:
        return 100
    span=max(upper-lower,1.0)
    distance=lower-value if value<lower else value-upper
    return max(0,round(100-100*distance/span))


def _location_matches(profile_regions,location):
    requested={
        part.strip().casefold()
        for part in location.split(",")
        if part.strip()
    }
    regions={str(region).strip().casefold() for region in profile_regions}
    if regions.intersection({"*","any","all regions"}):
        return 60
    if regions.intersection(requested):
        return 100
    return None


def resolve_season(selection,month):
    if selection!="Auto (India calendar; verify locally)":
        return selection
    if month in {6,7,8,9,10}:
        return "Kharif"
    if month in {11,12,1,2}:
        return "Rabi"
    return "Zaid"


def recommend_crops(soil_class,location,season,irrigation,climate,profiles):
    if any(not isinstance(value,str) for value in (soil_class,location,season,irrigation)):
        raise ValueError("Soil class, location, season, and irrigation selection must be text.")
    if not profiles:
        return {
            "recommendations":[],
            "reason":(
                "No top-five crop list is shown because this project has no sourced, "
                "region- and season-specific crop profiles configured."
            ),
        }
    if not location.strip():
        return {
            "recommendations":[],
            "reason":"Enter a district/state so the crop profiles can be region-matched.",
        }
    if irrigation.strip().casefold() in {"unknown","not specified",""}:
        return {
            "recommendations":[],
            "reason":"Select the irrigation availability before ranking crops.",
        }
    if not isinstance(climate,dict):
        return {"recommendations":[],"reason":"Seasonal climate values are required for crop screening."}
    try:
        temperature=float(climate["seasonal_mean_temperature_c"])
        rainfall=float(climate["seasonal_rainfall_mm"])
    except (KeyError,TypeError,ValueError):
        return {
            "recommendations":[],
            "reason":(
                "Enter a sourced seasonal mean temperature, seasonal rainfall estimate, and source. "
                "A current or 7-day forecast is not a seasonal climate normal."
            ),
        }
    climate_source=climate.get("source")
    if not isinstance(climate_source,str):
        return {
            "recommendations":[],
            "reason":"Provide the source for seasonal temperature and rainfall before ranking crops.",
        }
    climate_source=climate_source.strip()
    if not climate_source:
        return {
            "recommendations":[],
            "reason":"Provide the source for seasonal temperature and rainfall before ranking crops.",
        }
    if (
        not math.isfinite(temperature)
        or not -10<=temperature<=60
        or not math.isfinite(rainfall)
        or not 0<=rainfall<=10000
    ):
        raise ValueError("Seasonal climate values are outside the supported ranges.")
    ranked=[]
    for profile in profiles:
        soil_values={str(value).strip().casefold() for value in profile["soil_classes"]}
        soil=100 if soil_class.casefold() in soil_values else (
            60 if soil_values.intersection({"*","any"}) else None
        )
        if soil is None:
            continue
        region=_location_matches(profile["regions"],location)
        if region is None:
            continue
        profile_seasons={str(value).strip().casefold() for value in profile["seasons"]}
        season_score=(
            100 if season.casefold() in profile_seasons else
            75 if profile_seasons.intersection({"year-round","year round","all year"}) else
            None
        )
        if season_score is None:
            continue
        irrigation_options={str(value).strip().casefold() for value in profile["irrigation_options"]}
        irrigation_key=irrigation.strip().casefold()
        irrigation_score=(
            100 if irrigation_key in irrigation_options else
            70 if irrigation_options.intersection({"any","*","all"}) else
            None
        )
        if irrigation_score is None:
            continue
        temperature_score=_range_score(temperature,profile["temperature_c_range"])
        rainfall_score=_range_score(rainfall,profile["seasonal_rainfall_mm_range"])
        if temperature_score==0 or rainfall_score==0:
            continue
        components={
            "soil":soil,
            "region":region,
            "season":season_score,
            "temperature":temperature_score,
            "rainfall":rainfall_score,
            "irrigation":irrigation_score,
        }
        score=round(sum(_SCORE_WEIGHTS[key]*value for key,value in components.items()))
        ranked.append({
            "crop":profile["crop"],
            "score":score,
            "why":profile["why"],
            "growing_conditions":profile["growing_conditions"],
            "water_requirement":profile["water_requirement"],
            "improvements":profile["improvements"],
            "cultivation":profile["cultivation"],
            "components":components,
            "sources":profile["sources"],
            "climate_source":climate_source,
        })
    ranked.sort(key=lambda item:(-item["score"],item["crop"].casefold()))
    if not ranked:
        return {
            "recommendations":[],
            "reason":(
                "No configured crop profile matches this soil class, location, season, "
                "seasonal climate, and irrigation choice."
            ),
        }
    return {
        "recommendations":ranked[:5],
        "reason":"Rule-based suitability scores from configured sourced profiles; not yield predictions.",
    }
