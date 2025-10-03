import os
import csv
import time
import json
import re
import requests
import asyncio
import aiohttp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

class WeatherAgent:
    def __init__(self, zip_file="Zip_Codes.csv"):
        self.zip_codes = self.load_zip_codes(zip_file)
        self.memory = []  # short-term memory
        self.goals = [
            "Detect weather anomalies or rapid changes.",
            "Alert when weather varies significantly across regions.",
            "Track and visualize trends over time."
        ]
        self.last_weather_data = None
        self.check_interval = 60
        self.cycle_count = 0
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    # 1. Load ZIPs
    def load_zip_codes(self, filename):
        zip_codes = []
        with open(filename, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                zip_code = row.get("ZIPCODE") or row.get("ZIP_CODE_TEXT")
                if zip_code and zip_code.strip().isdigit():
                    zip_codes.append(zip_code.strip())
        return sorted(set(zip_codes))

    # 2. Async fetch
    async def fetch_weather(self, session, zip_code):
        url = f"http://api.openweathermap.org/data/2.5/weather?zip={zip_code},us&appid={WEATHER_API_KEY}&units=metric"
        try:
            async with session.get(url, timeout=10) as response:
                if response.status != 200:
                    return None
                data = await response.json()
                return {
                    "Zip Code": zip_code,
                    "Description": data["weather"][0]["description"],
                    "Temperature (°C)": data["main"]["temp"],
                    "Min Temp (°C)": data["main"]["temp_min"],
                    "Max Temp (°C)": data["main"]["temp_max"]
                }
        except:
            return None

    async def get_weather_data(self):
        connector = aiohttp.TCPConnector(limit_per_host=20)
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [self.fetch_weather(session, z) for z in self.zip_codes]
            results = await asyncio.gather(*tasks)
        return [r for r in results if r]

    # 3. Save to CSV
    def save_csv(self, data, filename):
        if not data: return
        with open(filename, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

    # 4. AI summary
    def generate_summary(self, df: pd.DataFrame):
        sample_data = df.head(20).to_string(index=False)
        prompt_text = (
            "You are a helpful assistant summarizing weather patterns.\n\n"
            f"{sample_data}\n\n"
            "Summarize:\n"
            "- Average, min, max temperatures\n"
            "- Most common conditions\n"
            "- Any outliers or interesting patterns"
        )
        prompt = ChatPromptTemplate.from_template("{input}")
        result = self.llm(prompt.format_prompt(input=prompt_text).to_messages())
        return result.content.strip()

    # 5. Discord
    def send_to_discord(self, content, file_path=None):
        if not DISCORD_WEBHOOK_URL:
            print("Webhook not set.")
            return
        files = {}
        if file_path and os.path.exists(file_path):
            with open(file_path, "rb") as f:
                files["file"] = (os.path.basename(file_path), f)
                response = requests.post(DISCORD_WEBHOOK_URL, data={"content": content}, files=files)
        else:
            response = requests.post(DISCORD_WEBHOOK_URL, data={"content": content})
        if response.ok:
            print("✅ Sent to Discord.")

    # 6. Analyze + decide
    def ai_decision(self, current_data):
        if not current_data: return {"action": "wait", "interval": 60}
        curr_df = pd.DataFrame(current_data)
        current_avg = curr_df["Temperature (°C)"].mean()
        context = f"""
Goal: {', '.join(self.goals)}

Current Data:
- Avg Temp: {round(current_avg,2)}°C
- Conditions: {curr_df['Description'].mode()[0] if not curr_df['Description'].mode().empty else 'N/A'}

{f"Previous Avg: {round(pd.DataFrame(self.last_weather_data)['Temperature (°C)'].mean(),2)}°C" if self.last_weather_data else ""}
Instructions:
Based on the data and goals, decide the situation.

Respond in JSON:
{{
  "situation": "...",
  "action": "...",
  "interval": 60,
  "reason": "...",
  "notable_observations": [...]
}}
        """
        result = self.llm.invoke(context)
        content = result.content.strip()
        try:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            return json.loads(match.group()) if match else {"action": "wait", "interval": 60}
        except:
            return {"action": "wait", "interval": 60}

    # 7. Log trends
    def log_trends(self, data, filename="weather_trends.csv"):
        df = pd.DataFrame(data)
        stats = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "avg_temp": round(df["Temperature (°C)"].mean(), 2),
            "avg_min_temp": round(df["Min Temp (°C)"].mean(), 2),
            "avg_max_temp": round(df["Max Temp (°C)"].mean(), 2),
            "most_common_condition": df["Description"].mode()[0] if not df["Description"].mode().empty else "N/A"
        }
        exists = os.path.isfile(filename)
        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(stats)

    # 8. Plot
    def plot_trends(self, trends_file="weather_trends.csv", plot_file="weather_plot.png"):
        if not os.path.isfile(trends_file): return
        df = pd.read_csv(trends_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['avg_temp'], label='Avg Temp', marker='o')
        plt.plot(df['timestamp'], df['avg_min_temp'], label='Min Temp', linestyle='--')
        plt.plot(df['timestamp'], df['avg_max_temp'], label='Max Temp', linestyle='--')
        plt.legend()
        plt.title("Temperature Trends Over Time")
        plt.xlabel("Time")
        plt.ylabel("°C")
        plt.grid(True)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.savefig(plot_file)
        print(f"📈 Trend plot saved to {plot_file}")
        return plot_file

    # 9. Main agent loop
    def run(self):
        print(f"🚀 Monitoring {len(self.zip_codes)} ZIP codes")
        try:
            while True:
                self.cycle_count += 1
                print(f"\n🔄 Cycle #{self.cycle_count} at {datetime.now().strftime('%H:%M:%S')}")
                weather_data = asyncio.run(self.get_weather_data())
                print(f"✅ Fetched data from {len(weather_data)} locations")

                # Decide next action
                decision = self.ai_decision(weather_data)
                print(f"🎯 AI Decision: {decision['action']} ({decision.get('situation', 'n/a')})")
                print(f"📝 Reason: {decision.get('reason', '')}")

                # Save + summarize if needed
                output_file = "weather_output.csv"
                self.save_csv(weather_data, output_file)

                if decision["action"] in ["alert", "investigate"]:
                    df = pd.DataFrame(weather_data)
                    summary = self.generate_summary(df)
                    self.send_to_discord(f"📢 **AI Summary**\n{summary}", output_file)

                # Log + plot
                self.log_trends(weather_data)
                plot_path = self.plot_trends()
                if decision["action"] == "alert":
                    self.send_to_discord("📊 Temperature trend alert!", plot_path)

                # Update memory
                self.last_weather_data = weather_data

                # Wait
                self.check_interval = decision.get("interval", 60)
                print(f"⏳ Waiting {self.check_interval} seconds...")
                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user.")

# === RUN ===
if __name__ == "__main__":
    agent = WeatherAgent()
    agent.run()
