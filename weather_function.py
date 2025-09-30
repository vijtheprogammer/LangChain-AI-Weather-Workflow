import requests
import os

def get_weather_forecast(zip_code):
    api_key = os.getenv("WEATHER_API_KEY")
    if not api_key:
        return "API key not found."

    url = f"http://api.openweathermap.org/data/2.5/weather?zip={zip_code},us&appid={api_key}&units=metric"
    response = requests.get(url)

    if response.status_code != 200:
        return f"Error fetching weather data for {zip_code}: {response.status_code}"

    data = response.json()
    weather_desc = data["weather"][0]["description"]
    temp = data["main"]["temp"]
    temp_min = data["main"]["temp_min"]
    temp_max = data["main"]["temp_max"]

    return f"The weather in {zip_code} is {weather_desc} with a temperature of {temp}°C (min: {temp_min}°C, max: {temp_max}°C)."

def get_weather_for_multiple_zips(zip_codes):
    results = []
    for z in zip_codes:
        weather = get_weather_forecast(z)
        results.append(weather)
    return "\n".join(results)
