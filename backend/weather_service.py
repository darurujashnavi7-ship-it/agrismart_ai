import requests

WINDY_THRESHOLD_KMH=30


def weather_condition(code):
    conditions={
        0:("☀️","Clear sky"),
        1:("🌤️","Mainly clear"),
        2:("⛅","Partly cloudy"),
        3:("☁️","Overcast"),
        45:("🌫️","Fog"),
        48:("🌫️","Depositing rime fog"),
        51:("🌦️","Light drizzle"),
        53:("🌦️","Moderate drizzle"),
        55:("🌧️","Dense drizzle"),
        56:("🌧️","Freezing drizzle"),
        57:("🌧️","Dense freezing drizzle"),
        61:("🌧️","Slight rain"),
        63:("🌧️","Moderate rain"),
        65:("🌧️","Heavy rain"),
        66:("🌧️","Freezing rain"),
        67:("🌧️","Heavy freezing rain"),
        71:("🌨️","Slight snow"),
        73:("🌨️","Moderate snow"),
        75:("❄️","Heavy snow"),
        77:("❄️","Snow grains"),
        80:("🌦️","Slight rain showers"),
        81:("🌧️","Moderate rain showers"),
        82:("⛈️","Violent rain showers"),
        85:("🌨️","Slight snow showers"),
        86:("❄️","Heavy snow showers"),
        95:("⛈️","Thunderstorm"),
        96:("⛈️","Thunderstorm with hail"),
        99:("⛈️","Thunderstorm with heavy hail"),
    }
    return conditions.get(code,("🌡️","Conditions unavailable"))


def weather_background_theme(data,wind_threshold_kmh=None):
    if wind_threshold_kmh is None:
        wind_threshold_kmh=WINDY_THRESHOLD_KMH

    current=data.get("current",{})
    code=current.get("weather_code")
    try:
        code=int(code)
    except (TypeError,ValueError):
        code=None

    try:
        precipitation=float(current.get("precipitation") or 0)
    except (TypeError,ValueError):
        precipitation=0
    try:
        wind=float(current.get("wind_speed_10m") or 0)
    except (TypeError,ValueError):
        wind=0

    if code in {95,96,99}:
        return "stormy"
    if code in {51,53,55,56,57,61,63,65,66,67,71,73,75,77,80,81,82,85,86} or precipitation>0:
        return "rainy"
    if code in {45,48}:
        return "foggy"
    if wind>wind_threshold_kmh:
        return "windy"
    if code in {1,2,3}:
        return "cloudy"
    if code==0:
        return "sunny"
    return "default"


def get_weather(place):
    geocoding=requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name":place,"count":1,"language":"en","format":"json"},
        timeout=20,
    )
    geocoding.raise_for_status()
    locations=geocoding.json().get("results",[])
    if not locations:
        raise ValueError("Location not found. Try a district or city.")

    location=locations[0]
    params={
        "latitude":location["latitude"],
        "longitude":location["longitude"],
        "current":(
            "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,"
            "weather_code,pressure_msl,surface_pressure,wind_speed_10m"
        ),
        "hourly":(
            "temperature_2m,precipitation_probability,precipitation,weather_code,"
            "wind_speed_10m,visibility"
        ),
        "daily":(
            "weather_code,temperature_2m_max,temperature_2m_min,uv_index_max,"
            "sunrise,sunset,precipitation_probability_max,precipitation_sum"
        ),
        "timezone":"auto",
        "forecast_days":7,
    }
    forecast=requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params=params,
        timeout=20,
    )
    forecast.raise_for_status()
    return location,forecast.json()


def farming_advice(data):
    current=data.get("current",{})
    daily=data.get("daily",{})
    rain_chance=max(daily.get("precipitation_probability_max") or [0])
    rainfall=max(daily.get("precipitation_sum") or [0])
    wind=current.get("wind_speed_10m",0)
    humidity=current.get("relative_humidity_2m",0)
    temperature=current.get("temperature_2m",0)
    uv_index=max(daily.get("uv_index_max") or [0])
    advice=[]

    if rain_chance>=60 or rainfall>=10:
        advice.append("Rain is likely. Check field drainage, delay spraying before rainfall, and avoid unnecessary irrigation.")
    if wind>=15:
        advice.append("Wind is strong. Avoid spraying in gusty conditions to reduce drift; secure lightweight covers and equipment.")
    if humidity>=80:
        advice.append("Humidity is high. Monitor crops for fungal symptoms, improve airflow, and avoid late-evening overhead irrigation.")
    if temperature>=35:
        advice.append("High temperatures are forecast. Schedule field work for cooler hours and check soil moisture before irrigating.")
    if uv_index>=8:
        advice.append("UV levels are very high. Plan outdoor farm work for early morning or late afternoon and use sun protection.")
    if not advice:
        advice.append("Check root-zone soil moisture before irrigation. Plan field work around the hourly forecast and follow product-label guidance.")
    return advice
