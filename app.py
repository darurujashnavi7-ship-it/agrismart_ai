import streamlit as st
from pathlib import Path
import hashlib
import json
import sys
from datetime import datetime

import requests
import streamlit.components.v1 as components

ROOT=Path(__file__).resolve().parent
LOGIN_BACKGROUND_URL="/app/static/login_background.png"
sys.path.insert(0,str(ROOT))
from backend.auth import init_db,register,login,get_profile,update_profile,add_history,get_history,delete_history
from backend.disease_detection import predict,model_status,get_supported_crops
from backend.fertilizer_recommendation import get_fertilizer_names
from backend.yield_prediction import predict_yield
from backend.weather_service import (
    WINDY_THRESHOLD_KMH,
    farming_advice,
    get_weather,
    weather_background_theme,
    weather_condition,
)
from backend.chatbot import LANGUAGES,answer,ChatbotError,is_ai_configured,transcribe_audio
from backend.soil_image_classifier import (
    model_status as soil_image_model_status,
    predict_soil_image,
    prepare_soil_image,
)
from backend.soil_image_guidance import (
    load_soil_guidance,
    soil_image_assistant_fallback,
    soil_image_assistant_context,
)
from backend.soil_crop_recommendations import (
    load_crop_profiles as load_soil_crop_profiles,
    recommend_crops,
    resolve_season,
)
from backend.soil_image_reporting import create_soil_image_report_pdf
st.set_page_config(page_title="AgriSmart | Precision Agriculture",page_icon="🌱",layout="wide")
init_db()
st.markdown("""<style>
.stApp{background:linear-gradient(135deg,#f3faf3,#fff 58%,#edf7ee)}
[data-testid="stSidebar"]{background:#e8f4e8}
.hero{padding:1.35rem 1.6rem;border-radius:18px;background:linear-gradient(110deg,#14532d,#26934b);color:white}
.hero h1{color:white;margin:0}.hero p{color:#e4f6e8;margin:.35rem 0 0}
.service-card{min-height:200px}
.service-card h3{margin:.25rem 0 .4rem}
.small{color:#53705a;font-size:.9rem}
[data-testid="stAppViewContainer"]:has(.agri-weather-scene){
    position:relative;isolation:isolate;background:transparent!important
}
[data-testid="stAppViewContainer"]:has(.agri-weather-scene)::before{
    content:"";position:fixed;inset:0;z-index:-2;
    background-color:#477843;
    background-image:linear-gradient(rgba(255,255,255,.12),rgba(7,38,19,.24)),url("/app/static/login_background.png");
    background-size:cover;background-position:center;
    animation:weather-scene-sunny .9s ease both
}
[data-testid="stAppViewContainer"]:has(.agri-weather-scene--cloudy)::before{
    background-image:linear-gradient(rgba(45,55,62,.42),rgba(42,57,58,.38)),url("/app/static/login_background.png");
    animation-name:weather-scene-cloudy
}
[data-testid="stAppViewContainer"]:has(.agri-weather-scene--rainy)::before{
    background-image:linear-gradient(rgba(24,49,66,.48),rgba(23,47,54,.44)),url("/app/static/login_background.png");
    animation-name:weather-scene-rainy
}
[data-testid="stAppViewContainer"]:has(.agri-weather-scene--windy)::before{
    background-image:linear-gradient(rgba(92,118,123,.32),rgba(45,86,67,.3)),url("/app/static/login_background.png");
    animation-name:weather-scene-windy
}
[data-testid="stAppViewContainer"]:has(.agri-weather-scene--stormy)::before{
    background-image:linear-gradient(rgba(14,22,39,.72),rgba(22,35,47,.65)),url("/app/static/login_background.png");
    animation-name:weather-scene-stormy
}
[data-testid="stAppViewContainer"]:has(.agri-weather-scene--foggy)::before{
    background-image:linear-gradient(rgba(198,210,207,.58),rgba(151,177,170,.56)),url("/app/static/login_background.png");
    animation-name:weather-scene-foggy
}
[data-testid="stAppViewContainer"]:has(.agri-weather-scene--default)::before{
    animation-name:weather-scene-default
}
[data-testid="stAppViewContainer"]:has(.agri-weather-scene--rainy)::after,
[data-testid="stAppViewContainer"]:has(.agri-weather-scene--stormy)::after{
    content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;
    background-image:repeating-linear-gradient(108deg,transparent 0 38px,rgba(230,244,255,.3) 39px,transparent 41px);
    animation:weather-rain .65s linear infinite
}
[data-testid="stAppViewContainer"]:has(.agri-weather-scene--stormy)::after{
    background-image:linear-gradient(116deg,transparent 0 48%,rgba(240,247,255,.88) 48.2%,transparent 48.8%),repeating-linear-gradient(108deg,transparent 0 38px,rgba(230,244,255,.3) 39px,transparent 41px);
    animation:weather-rain .65s linear infinite,weather-lightning 8s linear infinite
}
[data-testid="stAppViewContainer"]:has(.agri-weather-scene--windy)::after,
[data-testid="stAppViewContainer"]:has(.agri-weather-scene--cloudy)::after{
    content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;
    background-image:radial-gradient(ellipse at 20% 20%,rgba(255,255,255,.42),transparent 40%),radial-gradient(ellipse at 80% 35%,rgba(255,255,255,.32),transparent 34%);
    animation:weather-cloud-drift 24s ease-in-out infinite alternate
}
[data-testid="stAppViewContainer"]:has(.agri-weather-scene--foggy)::after{
    content:"";position:fixed;inset:0;z-index:-1;pointer-events:none;
    background:linear-gradient(180deg,rgba(239,245,239,.36),rgba(231,239,233,.62));
    animation:weather-fog-drift 12s ease-in-out infinite alternate
}
[data-testid="stAppViewContainer"]:has(.agri-weather-scene) [data-testid="stMetric"],
[data-testid="stAppViewContainer"]:has(.agri-weather-scene) [data-testid="stAlert"],
[data-testid="stAppViewContainer"]:has(.agri-weather-scene) [data-testid="stDataFrame"]{
    background:rgba(255,255,255,.84);backdrop-filter:blur(12px);
    border:1px solid rgba(255,255,255,.72);border-radius:14px
}
@keyframes weather-scene-sunny{from{opacity:.55}to{opacity:1}}
@keyframes weather-scene-cloudy{from{opacity:.55}to{opacity:1}}
@keyframes weather-scene-rainy{from{opacity:.55}to{opacity:1}}
@keyframes weather-scene-windy{from{opacity:.55}to{opacity:1}}
@keyframes weather-scene-stormy{from{opacity:.55}to{opacity:1}}
@keyframes weather-scene-foggy{from{opacity:.55}to{opacity:1}}
@keyframes weather-scene-default{from{opacity:.55}to{opacity:1}}
@keyframes weather-rain{from{background-position:0 0}to{background-position:-45px 90px}}
@keyframes weather-cloud-drift{from{transform:translateX(-1.5%)}to{transform:translateX(1.5%)}}
@keyframes weather-fog-drift{from{opacity:.55}to{opacity:.92}}
@keyframes weather-lightning{0%,91%,94%,100%{opacity:0}92%,93%{opacity:.8}}
@media(max-width:700px){
    [data-testid="stAppViewContainer"]:has(.agri-weather-scene)::before{background-attachment:scroll;background-position:center}
}
@media(prefers-reduced-motion:reduce){
    [data-testid="stAppViewContainer"]:has(.agri-weather-scene)::before,
    [data-testid="stAppViewContainer"]:has(.agri-weather-scene)::after{animation:none}
}
</style>""",unsafe_allow_html=True)

if "user" not in st.session_state:
    st.session_state.user=None

if not st.session_state.user:
    st.markdown(f"""
    <style>
    .stApp{{background:transparent!important}}
    [data-testid="stAppViewContainer"]{{
        background-image:linear-gradient(90deg,rgba(5,24,12,.58),rgba(5,24,12,.20)),url("{LOGIN_BACKGROUND_URL}");
        background-size:cover;background-position:center;background-attachment:fixed;
    }}
    [data-testid="stHeader"]{{background:rgba(0,0,0,0)}}
    [data-testid="stMainBlockContainer"]{{max-width:1180px;padding-top:4rem}}
    [data-testid="stVerticalBlockBorderWrapper"]{{
        background:rgba(255,255,255,.95);border:1px solid rgba(255,255,255,.75);
        border-radius:18px;padding:1.2rem;
    }}
    .login-brand{{min-height:520px;display:flex;flex-direction:column;justify-content:center;padding:2rem;color:white;text-shadow:0 2px 12px rgba(0,0,0,.65)}}
    .login-brand h1{{font-size:3.2rem;margin-bottom:.6rem}}
    .login-brand p{{font-size:1.2rem;max-width:34rem}}
    @media(max-width:900px){{
        [data-testid="stMainBlockContainer"]{{padding-top:1.2rem}}
        .login-brand{{min-height:220px;padding:1rem}}
        .login-brand h1{{font-size:2.4rem}}
        .login-brand p{{font-size:1rem}}
    }}
    </style>
    """,unsafe_allow_html=True)
    brand_column,form_column=st.columns([1.25,1],gap="large")
    with brand_column:
        st.markdown('<div class="login-brand"><h1>🌱 AgriSmart</h1><p>AI-powered precision agriculture for healthier crops and smarter decisions</p></div>',unsafe_allow_html=True)
    with form_column:
        with st.container(border=True):
            st.subheader("Welcome to AgriSmart")
            st.caption("Sign in or create a farmer account to continue.")
            tab1,tab2=st.tabs(["Login","Create account"])
            with tab1:
                username=st.text_input("Username",key="login_u")
                password=st.text_input("Password",type="password",key="login_p")
                if st.button("Login",type="primary",use_container_width=True):
                    if login(username,password):
                        st.session_state.user=username.strip().lower()
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
            with tab2:
                new_username=st.text_input("Choose username",key="reg_u")
                new_password=st.text_input("Choose password (minimum 8 characters)",type="password",key="reg_p")
                if st.button("Register",use_container_width=True):
                    ok,message=register(new_username,new_password)
                    (st.success if ok else st.error)(message)
            st.info("Create an account to access the farmer dashboard. Passwords are stored using salted PBKDF2 hashes.")
    st.stop()

user=st.session_state.user
district0,state0=get_profile(user)
service_pages=[
    "Dashboard",
    "Crop disease detection",
    "Crop yield prediction",
    "Weather recommendations",
    "Soil Analysis",
    "Ask AgriBot",
    "My history",
]
if st.session_state.get("active_page") not in service_pages:
    st.session_state.active_page="Dashboard"

with st.sidebar:
    st.title("🌱 AgriSmart")
    st.caption(f"Farmer: {user}")
    state=st.text_input("State",value=state0 or "Karnataka")
    district=st.text_input("District / town",value=district0 or "Bengaluru")
    if st.button("Save profile"):
        update_profile(user,district,state)
        st.success("Profile saved.")
    if st.button("Logout"):
        st.session_state.user=None
        st.session_state.pop("active_page",None)
        st.rerun()

page=st.session_state.active_page
page_subtitles={
    "Dashboard":"Detect • Predict • Plan • Learn",
    "Crop disease detection":"One crop-filtered diagnosis with fertilizer guidance",
    "Crop yield prediction":"Estimate crop yield and production",
    "Weather recommendations":"Local forecast and farming advice",
    "Soil Analysis":"Classify soil photos and review sourced crop guidance",
    "Ask AgriBot":"Agriculture questions and answers",
    "My history":"Your saved predictions and activity",
}
if page!="Dashboard" and st.button("← Dashboard",key="return_to_dashboard"):
    st.session_state.active_page="Dashboard"
    st.rerun()

st.markdown(
    f'<div class="hero"><h1>{page}</h1><p>{page_subtitles.get(page,"")}</p></div>',
    unsafe_allow_html=True,
)

if page=="Dashboard":
    st.caption("Your farming assistant for crop health, yield planning and weather-aware decisions.")
    st.subheader("Welcome to your farm dashboard")
    cards=[
        ("🔬","Crop disease detection","Identify possible leaf diseases from an image.","crop_disease_detection"),
        ("📈","Crop yield prediction","Estimate yield using crop and farm inputs.","crop_yield_prediction"),
        ("🌦️","Weather recommendations","Use local forecasts to plan farm activities.","weather_recommendations"),
        ("🧪","Soil Analysis","Classify soil photos and review crop suitability.","soil_analysis"),
        ("🤖","Ask AgriBot","Get help with farming questions.","ask_agribot"),
        ("📚","My history","Review your saved predictions and activity.","my_history"),
    ]
    for row_start in range(0,len(cards),3):
        columns=st.columns(3)
        for column,(icon,title,description,key) in zip(columns,cards[row_start:row_start+3]):
            with column:
                with st.container(border=True):
                    st.markdown(
                        f'<div class="service-card"><h2>{icon}</h2><h3>{title}</h3>'
                        f'<p class="small">{description}</p></div>',
                        unsafe_allow_html=True,
                    )
                    if st.button(f"Open {title}",key=f"open_{key}",use_container_width=True):
                        st.session_state.active_page=title
                        st.rerun()
    st.subheader("Getting started")
    st.write("Choose a service card to open it. For best disease results, upload a clear photo of one leaf in natural light.")

elif page=="Crop disease detection":
    st.subheader("AI crop disease detection")
    st.write("The model recognizes 38 PlantVillage leaf classes. Choose the crop and upload a close-up photo of one leaf; fruit and whole-plant photos are not covered.")
    if (
        st.session_state.get("soil_image_result")
        and st.session_state.get("soil_image_analysis_owner")==user
    ):
        st.caption(
            "The previous soil-photo classification is separate context; it is not used by the leaf disease model."
        )
    crop_options=get_supported_crops()
    selected_crop=st.selectbox("Crop in the image",["Select a crop..."]+crop_options)
    image=st.file_uploader("Upload leaf image",type=["jpg","jpeg","png"])
    if image:
        st.image(image,caption="Uploaded leaf image",width=360)
        if st.button("Detect disease",type="primary",disabled=selected_crop=="Select a crop..." or not model_status()):
            try:
                with st.spinner("Analyzing leaf image..."):
                    result=predict(image,expected_crop=selected_crop)
                st.metric("Prediction confidence",f"{result['confidence']*100:.1f}%")
                if not result["reliable"]:
                    st.warning("The model is unsure. Retake a close-up photo of one leaf from the selected crop and consult an agriculture expert.")
                else:
                    st.markdown("### Diagnosis result")
                    st.write(f"Crop: {result['crop']}")
                    st.write(f"Detected disease: {result['disease']}")
                    add_history(user,"Disease detection",f"{result['label']} ({result['confidence']:.3f})")
                st.markdown("### Fertilizer suggestion")
                fertilizer_label=result.get("raw_label",result["label"])
                try:
                    fertilizers=get_fertilizer_names(fertilizer_label)
                    st.write(", ".join(fertilizers))
                    st.caption("Crop-based guidance only; this is not a treatment for plant disease. Follow a soil test and local agriculture advice.")
                except ValueError as error:
                    st.info(str(error))
            except Exception as error:
                st.error(str(error))
    if not model_status():
        st.info("Model not found. Run the Colab notebook, then copy disease_model.keras and class_names.json into models/.")

elif page=="Crop yield prediction":
    st.subheader("Crop yield prediction")
    soil_image_result=st.session_state.get("soil_image_result")
    soil_crop_suggestion=None
    if (
        soil_image_result
        and st.session_state.get("soil_image_analysis_owner")==user
    ):
        image_prediction=soil_image_result["prediction"]
        if image_prediction["predicted_soil"]:
            st.info(
                f"Recent soil-photo class: {image_prediction['predicted_soil']} "
                f"({image_prediction['confidence']*100:.1f}% model score). "
                "This image result is not a measured soil property or a yield-model feature."
            )
        for crop_item in soil_image_result["crop_result"]["recommendations"]:
            if crop_item["crop"] in {
                "Rice","Wheat","Maize","Cotton","Sugarcane","Groundnut","Pulses",
                "Tomato","Potato","Chilli",
            }:
                soil_crop_suggestion=crop_item["crop"]
                break
    crop_options=["Rice","Wheat","Maize","Cotton","Sugarcane","Groundnut","Pulses","Tomato","Potato","Chilli"]
    crop_default=crop_options.index(soil_crop_suggestion) if soil_crop_suggestion in crop_options else 0
    if soil_crop_suggestion:
        st.caption(f"Selected crop suggestion from the configured soil profile: {soil_crop_suggestion}.")
    columns=st.columns(3)
    with columns[0]:
        crop=st.selectbox("Crop",crop_options,index=crop_default)
    with columns[1]:
        season=st.selectbox("Season",["Kharif","Rabi","Whole Year","Summer","Autumn","Winter"])
    with columns[2]:
        year=st.number_input("Crop year",min_value=2000,max_value=2035,value=2026)
    columns=st.columns(3)
    with columns[0]:
        area=st.number_input("Cultivated area (hectares)",min_value=.01,value=1.0)
    with columns[1]:
        rainfall=st.number_input("Annual rainfall (mm)",min_value=0.0,value=850.0)
    with columns[2]:
        fertilizer=st.number_input("Fertilizer use (kg/ha)",min_value=0.0,value=100.0)
    pesticide=st.number_input("Pesticide use (kg/ha; optional estimate)",min_value=0.0,value=1.0)
    if st.button("Predict crop yield",type="primary"):
        try:
            result=predict_yield(state,crop,season,year,area,rainfall,fertilizer,pesticide)
            st.metric("Estimated yield",f"{result['yield_t_ha']:.2f} tonnes/hectare")
            st.metric("Estimated total production",f"{result['production_tonnes']:.2f} tonnes")
            st.caption("Estimate based on historical patterns; actual harvest depends on local conditions.")
            add_history(user,"Yield prediction",f"{crop}: {result}")
        except Exception as error:
            st.error(str(error))

elif page=="Weather recommendations":
    st.subheader("Weather-aware farming recommendations")
    place=st.text_input("Enter district / town",value=district or "Bengaluru")
    if st.button("Get 7-day forecast",type="primary"):
        try:
            location,data=get_weather(place)
            current=data.get("current",{})
            import pandas as pd
            daily=data.get("daily",{})
            hourly=data.get("hourly",{})
            theme=weather_background_theme(data)
            st.markdown(
                f'<div class="agri-weather-scene agri-weather-scene--{theme}" aria-hidden="true"></div>',
                unsafe_allow_html=True,
            )
            icon,condition=weather_condition(current.get("weather_code"))
            location_name=", ".join(
                part for part in (
                    location.get("name"),
                    location.get("admin1"),
                    location.get("country"),
                ) if part
            )
            st.info(f"{icon} **{condition}** · {location_name}")
            columns=st.columns(4)
            columns[0].metric("Temperature",f"{current.get('temperature_2m','—')} °C")
            columns[1].metric("Feels like",f"{current.get('apparent_temperature','—')} °C")
            columns[2].metric("Humidity",f"{current.get('relative_humidity_2m','—')}%")
            columns[3].metric("Wind speed",f"{current.get('wind_speed_10m','—')} km/h")

            hourly_times=hourly.get("time",[])
            hour_index=next(
                (index for index,time in enumerate(hourly_times) if time>=current.get("time","")),
                0,
            )
            visibility_values=hourly.get("visibility",[])
            visibility=visibility_values[hour_index] if hour_index<len(visibility_values) else None
            visibility_label=f"{visibility/1000:.1f} km" if visibility is not None else "—"
            daily_values=lambda key: daily.get(key,[])
            uv_values=daily_values("uv_index_max")
            sunrise_values=daily_values("sunrise")
            sunset_values=daily_values("sunset")
            columns=st.columns(4)
            columns[0].metric("Rain now",f"{current.get('precipitation','—')} mm")
            columns[1].metric("Surface pressure",f"{current.get('surface_pressure','—')} hPa")
            columns[2].metric("Visibility",visibility_label)
            columns[3].metric("UV max today",uv_values[0] if uv_values else "—")
            sun_columns=st.columns(2)
            sun_columns[0].caption(f"Sunrise: {sunrise_values[0] if sunrise_values else '—'}")
            sun_columns[1].caption(f"Sunset: {sunset_values[0] if sunset_values else '—'}")

            st.subheader("Next 12 hours")
            hourly_fields={
                "time":hourly.get("time",[]),
                "temperature":hourly.get("temperature_2m",[]),
                "rain_chance":hourly.get("precipitation_probability",[]),
                "rain":hourly.get("precipitation",[]),
                "wind":hourly.get("wind_speed_10m",[]),
                "weather":hourly.get("weather_code",[]),
            }
            hourly_count=min((len(values) for values in hourly_fields.values()),default=0)
            selected_hours=range(hour_index,min(hour_index+12,hourly_count))
            hourly_frame=pd.DataFrame({
                "Time":[hourly_fields["time"][index][-5:] for index in selected_hours],
                "Weather":[weather_condition(hourly_fields["weather"][index])[1] for index in selected_hours],
                "Temp °C":[hourly_fields["temperature"][index] for index in selected_hours],
                "Rain chance %":[hourly_fields["rain_chance"][index] for index in selected_hours],
                "Rain mm":[hourly_fields["rain"][index] for index in selected_hours],
                "Wind km/h":[hourly_fields["wind"][index] for index in selected_hours],
            })
            if not hourly_frame.empty:
                chart_column,details_column=st.columns([1.1,1])
                with chart_column:
                    st.line_chart(hourly_frame.set_index("Time")[["Temp °C"]],height=240)
                with details_column:
                    st.dataframe(hourly_frame,use_container_width=True,hide_index=True)

            st.subheader("7-day forecast")
            daily_fields={
                "Date":daily.get("time",[]),
                "Weather code":daily.get("weather_code",[]),
                "Max °C":daily.get("temperature_2m_max",[]),
                "Min °C":daily.get("temperature_2m_min",[]),
                "UV max":daily.get("uv_index_max",[]),
                "Rain chance %":daily.get("precipitation_probability_max",[]),
                "Rain mm":daily.get("precipitation_sum",[]),
                "Sunrise":daily.get("sunrise",[]),
                "Sunset":daily.get("sunset",[]),
            }
            daily_count=min((len(values) for values in daily_fields.values()),default=0)
            frame=pd.DataFrame({
                "Date":daily_fields["Date"][:daily_count],
                "Weather":[weather_condition(code)[1] for code in daily_fields["Weather code"][:daily_count]],
                "Max °C":daily_fields["Max °C"][:daily_count],
                "Min °C":daily_fields["Min °C"][:daily_count],
                "UV max":daily_fields["UV max"][:daily_count],
                "Rain chance %":daily_fields["Rain chance %"][:daily_count],
                "Rain mm":daily_fields["Rain mm"][:daily_count],
                "Sunrise":daily_fields["Sunrise"][:daily_count],
                "Sunset":daily_fields["Sunset"][:daily_count],
            })
            if not frame.empty:
                st.line_chart(frame.set_index("Date")[["Max °C","Min °C"]],height=240)
                st.dataframe(frame,use_container_width=True,hide_index=True)
            st.subheader("Farm advisory")
            st.caption(f"Windy background activates above {WINDY_THRESHOLD_KMH} km/h.")
            with st.container(border=True):
                for recommendation in farming_advice(data):
                    st.write(f"- {recommendation}")
            st.caption(f"Source: Open-Meteo · Forecast timezone: {data.get('timezone','local time')}.")
        except Exception as error:
            st.error(f"Weather request failed: {error}")

elif page=="Soil Analysis":
    st.subheader("Soil Analysis & Crop Guidance")
    st.markdown("### AI soil image classification")
    st.caption(
        "Upload a clear soil-surface photograph. Image classification does not measure soil chemistry, "
        "moisture, fertility, or fertilizer requirements."
    )
    image_guidance=None
    try:
        image_guidance=load_soil_guidance()
    except ValueError as error:
        st.error(f"Soil image guidance is unavailable: {error}")
    crop_profile_config={"profiles":[]}
    crop_profile_error=None
    try:
        crop_profile_config=load_soil_crop_profiles()
    except ValueError as error:
        crop_profile_error=str(error)
        st.error(f"Crop screening is unavailable: {error}")
    if not soil_image_model_status():
        st.warning(
            "The trained soil image model is not installed yet. Run "
            "notebooks/Soil_Image_Classification_Colab.ipynb, then extract the model and JSON "
            "files from its ZIP into the project's models/ folder. No fallback or random prediction is used."
        )

    soil_photo=st.file_uploader(
        "Upload soil photo (JPG, JPEG, or PNG)",
        type=["jpg","jpeg","png"],
        key="soil_image_upload",
        help="Maximum 10 MB; minimum 128 × 128 pixels; upload a close, evenly lit view of soil.",
    )
    photo_bytes=soil_photo.getvalue() if soil_photo is not None else b""
    photo_signature=hashlib.sha256(photo_bytes).hexdigest() if photo_bytes else ""
    if photo_signature!=st.session_state.get("soil_image_signature",""):
        st.session_state["soil_image_result"]=None
        st.session_state["soil_image_analysis_owner"]=None
        st.session_state["soil_image_signature"]=photo_signature

    preview_image=None
    image_error=None
    if soil_photo is not None:
        try:
            preview_image=prepare_soil_image(soil_photo.name,photo_bytes)
            st.image(preview_image,caption="Uploaded soil photograph",use_container_width=True)
        except ValueError as error:
            image_error=str(error)
            st.error(image_error)

    farmer_columns=st.columns([1,2])
    image_farmer_name=farmer_columns[0].text_input(
        "Farmer name (optional)",
        max_chars=100,
        key="soil_image_farmer_name",
    )
    image_location=farmer_columns[1].text_input(
        "Farm district and state",
        value=", ".join(part for part in (district,state) if part),
        max_chars=200,
        key="soil_image_location",
    )
    climate_columns=st.columns(3)
    season_choice=climate_columns[0].selectbox(
        "Current growing season",
        ["Auto (India calendar; verify locally)","Kharif","Rabi","Zaid","Year-round"],
        key="soil_image_season",
    )
    irrigation_choice=climate_columns[1].selectbox(
        "Irrigation availability",
        ["Unknown","Rainfed","Supplemental","Assured"],
        key="soil_image_irrigation",
    )
    with climate_columns[2].expander("Seasonal climate inputs"):
        st.caption(
            "Use a local agricultural or long-term climate source. The live 7-day forecast "
            "is not a seasonal rainfall normal."
        )
        seasonal_temperature=st.number_input(
            "Seasonal mean temperature (°C)",
            min_value=-10.0,
            max_value=60.0,
            value=None,
            step=0.5,
            key="soil_image_seasonal_temperature",
        )
        seasonal_rainfall=st.number_input(
            "Seasonal rainfall estimate (mm)",
            min_value=0.0,
            max_value=10000.0,
            value=None,
            step=10.0,
            key="soil_image_seasonal_rainfall",
        )
        climate_source=st.text_input(
            "Source for seasonal climate values",
            max_chars=300,
            key="soil_image_climate_source",
        )
    analysis_input_signature=hashlib.sha256(json.dumps({
        "photo":photo_signature,
        "farmer":image_farmer_name,
        "location":image_location,
        "season":season_choice,
        "irrigation":irrigation_choice,
        "temperature":seasonal_temperature,
        "rainfall":seasonal_rainfall,
        "climate_source":climate_source,
    },sort_keys=True).encode("utf-8")).hexdigest()
    if analysis_input_signature!=st.session_state.get("soil_image_context_signature",""):
        st.session_state["soil_image_result"]=None
        st.session_state["soil_image_analysis_owner"]=None
        st.session_state["soil_image_context_signature"]=analysis_input_signature

    analyze_photo=st.button(
        "Analyze Soil Image",
        type="primary",
        disabled=soil_photo is None or preview_image is None or image_guidance is None,
        key="soil_image_analyze",
    )
    if analyze_photo:
        st.session_state["soil_image_result"]=None
        season=resolve_season(season_choice,datetime.now().month)
        seasonal_climate=(
            {
                "seasonal_mean_temperature_c":seasonal_temperature,
                "seasonal_rainfall_mm":seasonal_rainfall,
                "source":climate_source.strip(),
            }
            if seasonal_temperature is not None and seasonal_rainfall is not None
            else None
        )
        weather_summary={}
        weather_error=None
        try:
            with st.spinner("Checking the image and analyzing the soil category..."):
                prediction=predict_soil_image(soil_photo.name,photo_bytes)
                guidance_label=(
                    prediction["predicted_soil"]
                    or prediction["top_predictions"][0]["soil_type"]
                )
                soil_guidance=image_guidance["classes"].get(guidance_label)
                if soil_guidance is None:
                    raise ValueError(
                        f"No qualitative guidance is configured for model class {guidance_label}."
                    )
                if image_location.strip():
                    try:
                        weather_location,weather_data=get_weather(image_location.strip())
                        current_weather=weather_data.get("current",{})
                        daily_weather=weather_data.get("daily",{})
                        forecast_rain=sum(
                            float(value)
                            for value in (daily_weather.get("precipitation_sum") or [])
                            if value is not None
                        )
                        current_temperature=current_weather.get("temperature_2m")
                        current_condition=weather_condition(
                            current_weather.get("weather_code")
                        )[1]
                        weather_summary={
                            "place":weather_location.get("name",image_location.strip()),
                            "current_temperature_c":current_temperature,
                            "condition":current_condition,
                            "forecast_rain_next_7_days_mm":forecast_rain,
                            "summary":(
                                f"{current_condition}; current temperature "
                                f"{current_temperature if current_temperature is not None else 'unavailable'} °C; "
                                f"forecast rain {forecast_rain:.1f} mm over 7 days"
                            ),
                        }
                    except (requests.RequestException,ValueError,KeyError,TypeError) as error:
                        weather_error=f"Live weather context could not be loaded: {error}"
                if prediction["uncertain"]:
                    crop_result={
                        "recommendations":[],
                        "reason":"Crop screening is skipped because the soil image prediction is uncertain.",
                    }
                elif crop_profile_error:
                    crop_result={
                        "recommendations":[],
                        "reason":f"Crop profile configuration is invalid: {crop_profile_error}",
                    }
                else:
                    crop_result=recommend_crops(
                        prediction["predicted_soil"],
                        image_location,
                        season,
                        irrigation_choice,
                        seasonal_climate,
                        crop_profile_config["profiles"],
                    )
                image_result={
                    "prediction":prediction,
                    "guidance":soil_guidance,
                    "crop_result":crop_result,
                    "weather_summary":weather_summary,
                    "weather_error":weather_error,
                    "farmer_name":image_farmer_name,
                    "location":image_location,
                    "season":season,
                    "irrigation":irrigation_choice,
                    "climate_source":climate_source.strip(),
                    "file_name":soil_photo.name,
                    "image_bytes":photo_bytes,
                }
            st.session_state["soil_image_result"]=image_result
            st.session_state["soil_image_analysis_owner"]=user
            prediction_label=prediction["predicted_soil"] or "Uncertain"
            add_history(
                user,
                "Soil image classification",
                f"{prediction_label}; model score {prediction['confidence']:.3f}; {image_location}",
            )
        except FileNotFoundError as error:
            st.warning(str(error))
        except (ValueError,RuntimeError) as error:
            st.error(f"Soil image analysis failed: {error}")

    image_result=st.session_state.get("soil_image_result")
    if image_result and st.session_state.get("soil_image_analysis_owner")==user:
        st.caption("Results below match the last analyzed image and context. Reanalyze after changing inputs.")
        prediction=image_result["prediction"]
        crop_result=image_result["crop_result"]
        if prediction["uncertain"]:
            st.warning(
                "Uncertain result. Retake a clear, evenly lit soil-surface photo. This model only "
                "chooses among its trained labels and cannot reliably certify that an image is soil."
            )
        result_columns=st.columns(3)
        result_columns[0].metric(
            "Predicted soil category",
            prediction["predicted_soil"] or "Uncertain",
        )
        result_columns[1].metric("Top-class model score",f"{prediction['confidence']*100:.1f}%")
        test_accuracy=prediction.get("test_metrics",{}).get("accuracy")
        result_columns[2].metric(
            "Held-out test accuracy",
            f"{test_accuracy*100:.1f}%" if isinstance(test_accuracy,(int,float)) else "Not recorded",
        )
        st.progress(
            min(max(float(prediction["confidence"]),0.0),1.0),
            text="Top-class model score — not a calibrated probability",
        )
        st.caption(
            "Model label set: Alluvial, Arid, Black, Laterite, Mountain, Red, and Yellow Soil. "
            "Sandy, Clay, and Loamy image classes are not present in the selected dataset."
        )
        import plotly.graph_objects as go

        top_predictions=prediction["top_predictions"]
        prediction_figure=go.Figure(go.Bar(
            x=[item["confidence"]*100 for item in top_predictions],
            y=[item["soil_type"] for item in top_predictions],
            orientation="h",
            marker_color="#23864a",
            text=[f"{item['confidence']*100:.1f}%" for item in top_predictions],
            textposition="outside",
        ))
        prediction_figure.update_layout(
            title="Top 3 soil-class scores",
            xaxis={"title":"Model score (%)","range":[0,110]},
            yaxis={"autorange":"reversed"},
            template="plotly_white",
            height=260,
            margin=dict(l=20,r=35,t=55,b=20),
        )
        st.plotly_chart(prediction_figure,use_container_width=True)

        guidance=image_result["guidance"]
        if prediction["uncertain"]:
            st.caption("Crop examples are withheld because the soil image prediction is uncertain.")
        else:
            crop_guidance=guidance["crop_guidance"]
            st.subheader(f"Crops to consider for {prediction['predicted_soil']}")
            st.caption(
                "General examples from the cited source, not a local suitability score or guarantee. "
                "Confirm the crop against your district, season, rainfall, and irrigation."
            )
            if crop_guidance["crops"]:
                st.markdown("**Common crop examples:** "+", ".join(crop_guidance["crops"]))
            st.write(crop_guidance["note"])
            st.caption(
                f"Source: [{crop_guidance['source_title']}]({crop_guidance['source_url']})"
            )

        if crop_result["recommendations"] and not prediction["uncertain"]:
            st.subheader("Location-specific crop screening")
            st.caption(
                "These scores compare against configured local profiles; they are not probabilities "
                "or yield predictions."
            )
            if len(crop_result["recommendations"])<5:
                st.caption(
                    f"Only {len(crop_result['recommendations'])} sourced profiles matched; "
                    "the list is not padded with unsupported crops."
                )
            crop_columns=st.columns(2)
            for index,crop_item in enumerate(crop_result["recommendations"]):
                with crop_columns[index%2]:
                    with st.container(border=True):
                        st.markdown(f"**{index+1}. {crop_item['crop']} — {crop_item['score']}/100**")
                        st.progress(crop_item["score"])
                        st.write(crop_item["why"])
                        st.write(f"Seasonal conditions: {crop_item['growing_conditions']}")
                        st.write(f"Water requirement: {crop_item['water_requirement']}")
                        st.write("Soil improvements: "+"; ".join(crop_item["improvements"]))
                        st.write(f"Cultivation: {crop_item['cultivation']}")
                        for source in crop_item["sources"]:
                            st.caption(f"Source: {source['title']} — {source['url']}")

        image_pdf=create_soil_image_report_pdf(
            image_result["file_name"],
            image_result["image_bytes"],
            prediction,
            guidance,
            crop_result,
            image_result["farmer_name"],
            image_result["location"],
            image_result["season"],
            image_result["irrigation"],
            image_result["weather_summary"],
            image_result["climate_source"],
        )
        st.download_button(
            "Download Soil Image Analysis PDF",
            data=image_pdf,
            file_name="agrismart-soil-image-analysis.pdf",
            mime="application/pdf",
            key="soil_image_report_download",
        )

elif page=="Ask AgriBot":
    st.subheader("Ask AgriBot – your farming assistant")
    st.caption("Ask about crops, soil, irrigation, pests, plant diseases, weather, or farm planning. Include your crop and location for more useful advice. The bundled source excerpts cover IMD agricultural meteorological services.")
    if not is_ai_configured():
        st.info("Without GROQ_API_KEY, AgriBot can answer built-in FAQs and show relevant local source excerpts in English. A valid key is needed for open-ended multilingual answers and voice transcription.")
    if "chat" not in st.session_state:
        st.session_state.chat=[]
    if st.session_state.pop("clear_agribot_question",False):
        st.session_state.pop("agribot_question",None)

    language=st.selectbox(
        "Response language",
        list(LANGUAGES),
        key="agribot_language",
    )
    audio_file=st.audio_input("Ask AgriBot by voice",key="agribot_audio")
    if audio_file is not None:
        audio_bytes=audio_file.getvalue()
        audio_signature=hashlib.sha256(
            audio_bytes+language.encode("utf-8")
        ).hexdigest()
        if audio_signature!=st.session_state.get("agribot_voice_signature"):
            st.session_state["agribot_voice_signature"]=audio_signature
            try:
                with st.spinner("Transcribing your voice question..."):
                    st.session_state["agribot_question"]=transcribe_audio(
                        audio_bytes,
                        language=language,
                    )
                st.session_state["agribot_voice_error"]=None
            except ChatbotError as error:
                st.session_state["agribot_voice_error"]=str(error)
    if st.session_state.get("agribot_voice_error"):
        st.error(st.session_state["agribot_voice_error"])

    prompt=st.text_area(
        "Your question (you can edit the voice transcription)",
        max_chars=4000,
        key="agribot_question",
    )
    send_question=st.button("Send to AgriBot",type="primary")
    chat_languages=st.session_state.get("chat_languages",{})
    for message_index,(role,message) in enumerate(st.session_state.chat):
        with st.chat_message(role):
            st.write(message)
            if role=="assistant":
                spoken_text=json.dumps(message,ensure_ascii=False).replace("<","\\u003c")
                spoken_locale=json.dumps(
                    chat_languages.get(message_index,LANGUAGES[language]["locale"])
                )
                components.html(
                    f"""
                    <button onclick="speakAnswer()"
                        style="border:1px solid #cbd5ce;border-radius:16px;background:#fff;
                        color:#245b35;padding:5px 12px;cursor:pointer;font-size:14px">
                        🔊 Listen
                    </button>
                    <script>
                    const answerText={spoken_text};
                    const answerLocale={spoken_locale};
                    function speakAnswer(){{
                        if (!("speechSynthesis" in window)) {{
                            alert("Speech playback is not supported by this browser.");
                            return;
                        }}
                        window.speechSynthesis.cancel();
                        const utterance=new SpeechSynthesisUtterance(answerText);
                        utterance.lang=answerLocale;
                        window.speechSynthesis.speak(utterance);
                    }}
                    </script>
                    """,
                    height=42,
                )
    if send_question and prompt.strip():
        st.session_state.chat.append(("user",prompt))
        image_result=st.session_state.get("soil_image_result")
        if st.session_state.get("soil_image_analysis_owner")!=user:
            image_result=None
        local_soil_answer=(
            soil_image_assistant_fallback(
                prompt,
                image_result["prediction"],
                image_result["guidance"],
                image_result["crop_result"],
            )
            if image_result else None
        )
        try:
            assistant_context=""
            if image_result:
                assistant_context=soil_image_assistant_context(
                    image_result["prediction"],
                    image_result["guidance"],
                    image_result["crop_result"],
                    image_result["location"],
                    image_result["season"],
                    image_result["irrigation"],
                    image_result["climate_source"],
                    image_result["weather_summary"],
                )
            if local_soil_answer and not is_ai_configured():
                response=local_soil_answer
            else:
                response=answer(
                    prompt,
                    history=st.session_state.chat[:-1],
                    context=assistant_context,
                    language=language,
                )
        except ChatbotError as error:
            response=local_soil_answer or f"AgriBot could not answer: {error}"
        st.session_state.chat.append(("assistant",response))
        st.session_state.setdefault("chat_languages",{})[
            len(st.session_state.chat)-1
        ]=LANGUAGES[language]["locale"]
        st.session_state["clear_agribot_question"]=True
        st.rerun()
    st.caption("Use the response-language selector, record a voice question, and press Send. Use Listen under an answer to hear it with your browser's speech engine. Advice is general guidance, not a confirmed diagnosis or a substitute for local agricultural advice.")

elif page=="My history":
    st.subheader("Your recent activity")
    rows=get_history(user)
    if rows:
        headings=st.columns([1.2,3,1.5,.8])
        headings[0].caption("Module")
        headings[1].caption("Result")
        headings[2].caption("Date")
        for history_id,module,result,created_at in rows:
            columns=st.columns([1.2,3,1.5,.8])
            columns[0].write(module)
            columns[1].write(result)
            columns[2].caption(created_at)
            if columns[3].button("Delete",key=f"delete_history_{history_id}"):
                if delete_history(user,history_id):
                    st.success("History entry deleted.")
                    st.rerun()
                else:
                    st.error("History entry was not found or does not belong to this account.")
    else:
        st.info("No activity recorded yet.")
