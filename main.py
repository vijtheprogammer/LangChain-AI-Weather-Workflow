import os
import time
import csv
import requests
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# 1. Load ZIP codes
def load_zip_codes_from_csv(filename):
    zip_codes = []
    with open(filename, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            zip_code = row.get("ZIPCODE") or row.get("ZIP_CODE_TEXT")
            if zip_code and zip_code.strip().isdigit():
                zip_codes.append(zip_code.strip())
    return sorted(set(zip_codes))

# 2. Get weather data for a ZIP code
def get_weather_forecast_structured(zip_code):
    url = f"http://api.openweathermap.org/data/2.5/weather?zip={zip_code},us&appid={WEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return None
        data = response.json()
        return {
            "Zip Code": zip_code,
            "Description": data["weather"][0]["description"],
            "Temperature (°C)": data["main"]["temp"],
            "Min Temp (°C)": data["main"]["temp_min"],
            "Max Temp (°C)": data["main"]["temp_max"]
        }
    except Exception as e:
        print(f"Error fetching weather for {zip_code}: {e}")
        return None

# 3. Save data to CSV
def save_weather_data_to_csv(data, filename):
    if not data:
        print("No data to save.")
        return
    keys = data[0].keys()
    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(file, keys)
        writer.writeheader()
        writer.writerows(data)
    print(f"✅ Saved weather data to {filename}")

# 4. AI Summary using LangChain
def generate_summary_with_llm(weather_df: pd.DataFrame) -> str:
    # Format weather data into a string table for input
    sample_data = weather_df.head(20).to_string(index=False)

    prompt_text = (
        "You are a helpful assistant summarizing weather patterns.\n"
        "Here is a sample of weather data from various U.S. ZIP codes:\n\n"
        f"{sample_data}\n\n"
        "Please summarize the key findings in the following format:\n"
        "- Average, min, and max temperatures\n"
        "- Most common weather conditions\n"
        "- Any unusual outliers or patterns\n"
        "Be concise but insightful."
    )

    prompt = ChatPromptTemplate.from_template("{input}")

    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    result = llm(prompt.format_prompt(input=prompt_text).to_messages())
    return result.content.strip()


# 5. Send summary and CSV to Discord
def send_results_to_discord(summary_text, file_path):
    if not DISCORD_WEBHOOK_URL:
        print("DISCORD_WEBHOOK_URL not set.")
        return

    try:
        with open(file_path, 'rb') as file:
            payload = {
                "content": f"🤖 **AI Weather Summary:**\n{summary_text}"
            }
            files = {
                "file": (os.path.basename(file_path), file)
            }
            response = requests.post(DISCORD_WEBHOOK_URL, data=payload, files=files)
            response.raise_for_status()
            print("📤 Summary + CSV sent to Discord.")
    except Exception as e:
        print(f"Failed to send to Discord: {e}")

# 6. Main pipeline
def main():
    zip_codes = load_zip_codes_from_csv("Zip_Codes.csv")

    weather_data = []
    chunk_size = 50
    delay_seconds = 60

    total = len(zip_codes)
    processed = 0

    for i in range(0, total, chunk_size):
        batch = zip_codes[i:i + chunk_size]
        print(f"\n🔄 Processing ZIP codes {i + 1} to {i + len(batch)} of {total}...\n")
        for zip_code in batch:
            weather = get_weather_forecast_structured(zip_code)
            if weather:
                weather_data.append(weather)
            processed += 1
            print(f"  ✅ [{processed}/{total}] {zip_code}")
            time.sleep(1)
        if i + chunk_size < total:
            print(f"\n⏳ Waiting {delay_seconds} seconds before next batch...\n")
            time.sleep(delay_seconds)

    output_csv = "weather_output.csv"
    save_weather_data_to_csv(weather_data, output_csv)

    # Generate AI summary
    df = pd.DataFrame(weather_data)
    summary = generate_summary_with_llm(df)

    # Send everything to Discord
    send_results_to_discord(summary, output_csv)

if __name__ == "__main__":
    main()
