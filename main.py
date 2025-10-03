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
        self.memory = []
        self.goals = [
            "Detect weather anomalies or rapid changes.",
            "Alert when weather varies significantly across regions.",
            "Track and visualize trends over time."
        ]
        self.last_weather_data = None
        self.cycle_count = 0
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

    def load_zip_codes(self, filename):
        zip_codes = []
        with open(filename, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                zip_code = row.get("ZIPCODE") or row.get("ZIP_CODE_TEXT")
                if zip_code and zip_code.strip().isdigit():
                    zip_codes.append(zip_code.strip())
        return sorted(set(zip_codes))

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

    def save_csv(self, data, filename):
        if not data: return
        with open(filename, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

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

    def send_to_discord(self, content, file_path=None):
        if not DISCORD_WEBHOOK_URL:
            print("❌ Webhook not set.")
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
        else:
            print(f"❌ Failed to send to Discord: {response.status_code}")

    def ai_decision(self, current_data):
        if not current_data: 
            return {"action": "wait", "interval": 900, "reason": "No data fetched."}
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
  "interval": 900,
  "reason": "...",
  "notable_observations": []
}}
        """
        result = self.llm.invoke(context)
        content = result.content.strip()
        try:
            match = re.search(r"\{.*\}", content, re.DOTALL)
            return json.loads(match.group()) if match else {"action": "wait", "interval": 900, "reason": "No valid AI response."}
        except:
            return {"action": "wait", "interval": 900, "reason": "Failed to parse AI response."}

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

    def plot_trends(self, trends_file="weather_trends.csv", plot_file="weather_plot.png"):
        if not os.path.isfile(trends_file): 
            return None
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

    def run(self):
        print(f"🚀 Monitoring {len(self.zip_codes)} ZIP codes")
        routine_summary_interval = 4  # every ~1 hour if 15m interval

        try:
            while True:
                self.cycle_count += 1
                print(f"\n🔄 Cycle #{self.cycle_count} started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                weather_data = asyncio.run(self.get_weather_data())
                print(f"✅ Fetched data from {len(weather_data)} ZIP codes.")

                decision = self.ai_decision(weather_data)
                print(f"🤖 AI Decision: {decision.get('action', 'wait')} — {decision.get('situation', 'No situation provided')}")
                print(f"🧠 Reason: {decision.get('reason', 'No reason provided')}")

                output_file = "weather_output.csv"
                self.save_csv(weather_data, output_file)
                print(f"💾 Weather data saved to {output_file}")

                self.log_trends(weather_data)
                plot_path = self.plot_trends()

                send_detailed = decision['action'] in ['alert', 'investigate']
                send_routine = (self.cycle_count % routine_summary_interval == 0)

                if send_detailed:
                    print("📡 Sending detailed alert to Discord...")
                    df = pd.DataFrame(weather_data)
                    summary = self.generate_summary(df)
                    self.send_to_discord(f"🚨 **Weather Alert**\n{summary}", output_file)
                    if plot_path:
                        self.send_to_discord("📊 Weather trend update:", plot_path)

                elif send_routine:
                    print("ℹ️ Sending routine summary to Discord...")
                    df = pd.DataFrame(weather_data)
                    summary = self.generate_summary(df)
                    self.send_to_discord(f"ℹ️ **Routine Weather Summary**\n{summary}")

                else:
                    print("🔕 No Discord update this cycle (no alert, not time for summary).")

                self.last_weather_data = weather_data

                interval = decision.get('interval', 900)
                interval = max(300, min(900, interval))  # Clamp to 5–15 min
                print(f"⏳ Sleeping for {interval // 60} minutes...\n")
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user.")

# === RUN ===
if __name__ == "__main__":
    agent = WeatherAgent()
    agent.run()
