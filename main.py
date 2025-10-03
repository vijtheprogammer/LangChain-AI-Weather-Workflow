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
    def __init__(self, zip_file="US.txt"):
        self.zip_codes, self.zip_to_region = self.load_zip_codes(zip_file)
        self.memory = []  # short-term memory
        self.goals = [
            "Detect weather anomalies or rapid changes.",
            "Alert when weather varies significantly across regions.",
            "Track and visualize trends over time."
        ]
        self.last_weather_data = None
        self.check_interval = 1800  # 30 minutes default
        self.cycle_count = 0
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        self.concurrent_requests = 300

    # Load ZIPs and assign regions by state groups (Northeast, Midwest, etc.)
    def load_zip_codes(self, filename):
        zip_codes = []
        zip_to_region = {}
        # Define US regions by state abbreviations
        regions = {
            "Northeast": {"ME", "NH", "VT", "MA", "RI", "CT", "NY", "NJ", "PA"},
            "Midwest": {"OH", "MI", "IN", "IL", "WI", "MN", "IA", "MO", "ND", "SD", "NE", "KS"},
            "South": {"DE", "MD", "VA", "WV", "NC", "SC", "GA", "FL", "KY", "TN", "MS", "AL", "OK", "TX", "AR", "LA"},
            "West": {"ID", "MT", "WY", "NV", "UT", "CO", "AZ", "NM", "AK", "WA", "OR", "CA", "HI"}
        }

        with open(filename, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("\t")
                if len(parts) < 5:
                    continue
                country, zipcode, city, state_full, state_abbr = parts[:5]
                if country != "US" or not zipcode.isdigit():
                    continue
                zip_codes.append(zipcode)
                region = "Other"
                for rgn, states in regions.items():
                    if state_abbr in states:
                        region = rgn
                        break
                zip_to_region[zipcode] = region
        return sorted(set(zip_codes)), zip_to_region

    # Async fetch weather per ZIP code
    async def fetch_weather(self, session, zip_code):
        url = f"http://api.openweathermap.org/data/2.5/weather?zip={zip_code},us&appid={WEATHER_API_KEY}&units=metric"
        try:
            async with session.get(url, timeout=10) as response:
                if response.status == 404:
                    print(f"⚠️ Failed ZIP {zip_code}: HTTP 404 (Not Found)")
                    return None
                elif response.status != 200:
                    print(f"⚠️ Failed ZIP {zip_code}: HTTP {response.status}")
                    return None
                data = await response.json()
                return {
                    "Zip Code": zip_code,
                    "Region": self.zip_to_region.get(zip_code, "Other"),
                    "Description": data["weather"][0]["description"],
                    "Temperature (°C)": data["main"]["temp"],
                    "Min Temp (°C)": data["main"]["temp_min"],
                    "Max Temp (°C)": data["main"]["temp_max"]
                }
        except Exception as e:
            print(f"⚠️ Exception fetching ZIP {zip_code}: {e}")
            return None

    # Get weather data asynchronously with controlled concurrency
    async def get_weather_data(self):
        connector = aiohttp.TCPConnector(limit_per_host=self.concurrent_requests)
        timeout = aiohttp.ClientTimeout(total=300)  # 5 min timeout
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [self.fetch_weather(session, z) for z in self.zip_codes]
            results = []
            for i in range(0, len(tasks), self.concurrent_requests):
                batch = tasks[i:i+self.concurrent_requests]
                batch_results = await asyncio.gather(*batch)
                results.extend(batch_results)
                print(f"Fetched {len(results)} weather records so far...")
            return [r for r in results if r]

    # Save full weather data CSV
    def save_csv(self, data, filename):
        if not data: return
        with open(filename, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        print(f"💾 Saved CSV: {filename}")

    # AI summary of weather data (top 20 for brevity)
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
        result = self.llm.invoke(prompt.format_prompt(input=prompt_text).to_messages())
        return result.content.strip()

    # Send message (and optionally file) to Discord
    def send_to_discord(self, content, file_path=None):
        if not DISCORD_WEBHOOK_URL:
            print("Webhook not set.")
            return
        try:
            if file_path and os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    files = {"file": (os.path.basename(file_path), f)}
                    response = requests.post(DISCORD_WEBHOOK_URL, data={"content": content}, files=files)
            else:
                response = requests.post(DISCORD_WEBHOOK_URL, data={"content": content})
            if response.ok:
                print("✅ Sent to Discord.")
            else:
                print(f"⚠️ Discord webhook failed: {response.status_code} {response.text}")
        except Exception as e:
            print(f"⚠️ Exception sending Discord message: {e}")

    # Log temperature trends grouped by region for each run
    def log_trends_by_region(self, data, filename="weather_trends_by_region.csv"):
        df = pd.DataFrame(data)
        if df.empty:
            print("⚠️ No data to log trends.")
            return
        grouped = df.groupby("Region")
        rows = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for region, group in grouped:
            rows.append({
                "timestamp": now_str,
                "region": region,
                "avg_temp": round(group["Temperature (°C)"].mean(), 2),
                "avg_min_temp": round(group["Min Temp (°C)"].mean(), 2),
                "avg_max_temp": round(group["Max Temp (°C)"].mean(), 2),
                "most_common_condition": group["Description"].mode()[0] if not group["Description"].mode().empty else "N/A"
            })
        exists = os.path.isfile(filename)
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
        print(f"📊 Logged trends by region to {filename}")

    # Plot temperature trends by region with clean, larger figure
    def plot_trends_by_region(self, trends_file="weather_trends_by_region.csv", plot_file="weather_region_plot.png"):
        if not os.path.isfile(trends_file):
            print("⚠️ Trends file missing, skipping plot.")
            return None
        df = pd.read_csv(trends_file)

        # Robust timestamp parsing
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
        bad_ts_count = df["timestamp"].isna().sum()
        if bad_ts_count > 0:
            print(f"⚠️ Warning: Dropped {bad_ts_count} rows due to invalid timestamps in trends CSV.")
        df = df.dropna(subset=["timestamp"])

        plt.figure(figsize=(18, 10))
        regions = df["region"].unique()

        for region in regions:
            region_df = df[df["region"] == region].sort_values("timestamp")
            plt.plot(region_df["timestamp"], region_df["avg_temp"], marker='o', label=region)

        plt.legend(title="Region")
        plt.title("Average Temperature Trends by Region")
        plt.xlabel("Time")
        plt.ylabel("Temperature (°C)")
        plt.grid(True)
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()
        print(f"📈 Saved region trend plot: {plot_file}")
        return plot_file

    # Main agent loop
    def run(self):
        print(f"🚀 Monitoring {len(self.zip_codes)} ZIP codes.")
        try:
            while True:
                self.cycle_count += 1
                start_time = time.time()
                print(f"\n🔄 Cycle #{self.cycle_count} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                weather_data = asyncio.run(self.get_weather_data())
                print(f"✅ Fetched weather data for {len(weather_data)} ZIP codes.")

                if not weather_data:
                    print("⚠️ No weather data fetched this cycle, skipping.")
                    time.sleep(self.check_interval)
                    continue

                output_file = "weather_data_full.csv"
                self.save_csv(weather_data, output_file)

                self.log_trends_by_region(weather_data)
                plot_path = self.plot_trends_by_region()

                df = pd.DataFrame(weather_data)
                summary = self.generate_summary(df)
                discord_message = (
                    f"📢 **Weather Summary (Cycle {self.cycle_count})**\n"
                    f"Average Temperature: {round(df['Temperature (°C)'].mean(), 2)}°C\n"
                    f"Most Common Condition: {df['Description'].mode()[0] if not df['Description'].mode().empty else 'N/A'}\n\n"
                    f"**AI Summary:**\n{summary}"
                )

                self.send_to_discord(discord_message, plot_path)

                elapsed = time.time() - start_time
                sleep_time = max(0, self.check_interval - elapsed)
                print(f"⏳ Cycle took {elapsed:.2f}s, sleeping {sleep_time:.2f}s before next update...")
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user.")

# === RUN ===
if __name__ == "__main__":
    agent = WeatherAgent()
    agent.check_interval = 1800  # 30 minutes interval
    agent.run()
