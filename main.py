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

# Region to states mapping
REGIONS = {
    "Northeast": ['CT', 'ME', 'MA', 'NH', 'RI', 'VT', 'NJ', 'NY', 'PA'],
    "Midwest": ['IL', 'IN', 'MI', 'OH', 'WI', 'IA', 'KS', 'MN', 'MO', 'NE', 'ND', 'SD'],
    "South": ['DE', 'FL', 'GA', 'MD', 'NC', 'SC', 'VA', 'WV', 'AL', 'KY', 'MS', 'TN', 'AR', 'LA', 'OK', 'TX'],
    "West": ['AZ', 'CO', 'ID', 'MT', 'NV', 'NM', 'UT', 'WY', 'AK', 'CA', 'HI', 'OR', 'WA']
}

class WeatherAgent:
    def __init__(self, zip_file="US.txt"):
        self.zip_data = self.load_zip_data(zip_file)  # List of dicts with keys: zip, state
        self.memory = []  # short-term memory if needed
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
        self.cycle_count = 0
        self.check_interval = 1800  # 30 minutes

    def load_zip_data(self, filename):
        zip_list = []
        with open(filename, "r") as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                country, zip_code, city, state_full, state_code = parts[:5]
                if country == "US" and zip_code.isdigit():
                    zip_list.append({"zip": zip_code, "state": state_code})
        print(f"🗂️ Loaded {len(zip_list)} ZIP codes.")
        return zip_list

    # Async fetch for weather per ZIP (US only)
    async def fetch_weather(self, session, zip_info):
        zip_code = zip_info["zip"]
        url = f"http://api.openweathermap.org/data/2.5/weather?zip={zip_code},us&appid={WEATHER_API_KEY}&units=imperial"
        try:
            async with session.get(url, timeout=15) as resp:
                if resp.status != 200:
                    # print(f"⚠️ Failed ZIP {zip_code}: HTTP {resp.status}")
                    return None
                data = await resp.json()
                return {
                    "Zip Code": zip_code,
                    "State": zip_info["state"],
                    "Description": data["weather"][0]["description"],
                    "Temperature (°F)": data["main"]["temp"],
                    "Min Temp (°F)": data["main"]["temp_min"],
                    "Max Temp (°F)": data["main"]["temp_max"]
                }
        except Exception as e:
            # print(f"⚠️ Exception ZIP {zip_code}: {e}")
            return None

    async def get_all_weather(self, concurrency=300):
        connector = aiohttp.TCPConnector(limit_per_host=concurrency)
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [self.fetch_weather(session, zi) for zi in self.zip_data]
            results = []
            # Process in chunks with progress feedback
            for i in range(0, len(tasks), concurrency):
                chunk = tasks[i:i+concurrency]
                chunk_results = await asyncio.gather(*chunk)
                results.extend([r for r in chunk_results if r])
                print(f"📡 Fetched {len(results)} weather records so far...")
        return results

    def assign_region(self, state_code):
        for region, states in REGIONS.items():
            if state_code in states:
                return region
        return "Other"

    def save_csv(self, data, filename):
        if not data:
            print("⚠️ No data to save CSV.")
            return
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        print(f"💾 Saved CSV: {filename}")

    def log_trends_by_region(self, data, filename="weather_trends_by_region.csv"):
        if not data:
            print("⚠️ No data for trend logging.")
            return
        df = pd.DataFrame(data)
        df['Region'] = df['State'].map(self.assign_region)

        stats = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for region in df['Region'].unique():
            reg_df = df[df['Region'] == region]
            stats.append({
                "timestamp": timestamp,
                "region": region,
                "avg_temp": round(reg_df["Temperature (°F)"].mean(), 2),
                "min_temp": round(reg_df["Min Temp (°F)"].mean(), 2),
                "max_temp": round(reg_df["Max Temp (°F)"].mean(), 2),
                "most_common_condition": reg_df["Description"].mode()[0] if not reg_df["Description"].mode().empty else "N/A"
            })

        exists = os.path.isfile(filename)
        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=stats[0].keys())
            if not exists:
                writer.writeheader()
            writer.writerows(stats)
        print(f"📊 Logged trends by region to {filename}")

    def plot_trends_by_region(self, trends_file="weather_trends_by_region.csv", plot_file="weather_region_plot.png"):
        if not os.path.isfile(trends_file):
            print("⚠️ Trend file not found for plotting.")
            return None

        df = pd.read_csv(trends_file)
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception as e:
            print(f"⚠️ Error parsing timestamp: {e}")
            return None

        plt.figure(figsize=(16, 9))
        regions = df['region'].unique()
        for region in regions:
            reg_df = df[df['region'] == region].sort_values("timestamp")
            plt.plot(reg_df['timestamp'], reg_df['avg_temp'], label=region, marker='o')

        plt.title("Regional Average Temperature Trends (°F)")
        plt.xlabel("Time")
        plt.ylabel("Avg Temperature (°F)")
        plt.legend(title="Region")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()
        print(f"📈 Saved region trend plot: {plot_file}")
        return plot_file

    def generate_summary(self, data):
        if not data:
            return "⚠️ No data to summarize."

        df = pd.DataFrame(data)
        df['Region'] = df['State'].map(self.assign_region)

        summary_text = f"Weather Summary by Region as of {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        for region in sorted(df['Region'].unique()):
            reg_df = df[df['Region'] == region]
            avg_temp = reg_df["Temperature (°F)"].mean()
            min_temp = reg_df["Min Temp (°F)"].mean()
            max_temp = reg_df["Max Temp (°F)"].mean()
            common_cond = reg_df["Description"].mode()[0] if not reg_df["Description"].mode().empty else "N/A"

            summary_text += (
                f"**{region}**\n"
                f"- Avg Temp: {avg_temp:.1f} °F\n"
                f"- Min Temp: {min_temp:.1f} °F\n"
                f"- Max Temp: {max_temp:.1f} °F\n"
                f"- Most Common Condition: {common_cond}\n\n"
            )
        return summary_text.strip()

    def send_to_discord(self, content, file_paths=[]):
        if not DISCORD_WEBHOOK_URL:
            print("⚠️ Discord webhook URL not set.")
            return
        try:
            files = []
            for path in file_paths:
                if os.path.exists(path):
                    files.append(("file", (os.path.basename(path), open(path, "rb"))))
            payload = {"content": content}
            response = requests.post(DISCORD_WEBHOOK_URL, data=payload, files=files)
            if response.ok:
                print("✅ Sent update to Discord.")
            else:
                print(f"⚠️ Discord post failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"⚠️ Exception sending Discord message: {e}")

    def run(self):
        print(f"🚀 Starting WeatherAgent with {len(self.zip_data)} ZIP codes.")
        try:
            while True:
                self.cycle_count += 1
                start_time = time.time()
                print(f"\n🔄 Cycle #{self.cycle_count} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Fetch weather data async
                weather_data = asyncio.run(self.get_all_weather(concurrency=300))
                print(f"✅ Fetched weather data for {len(weather_data)} ZIP codes.")

                # Save full data CSV
                self.save_csv(weather_data, "weather_data_full.csv")

                # Log trends by region
                self.log_trends_by_region(weather_data)

                # Plot region trends
                plot_path = self.plot_trends_by_region()

                # Generate AI summary
                summary = self.generate_summary(weather_data)

                # Send summary + files to Discord
                files_to_send = ["weather_data_full.csv"]
                if plot_path:
                    files_to_send.append(plot_path)

                self.send_to_discord(summary, files_to_send)

                elapsed = time.time() - start_time
                sleep_time = max(self.check_interval - elapsed, 0)
                print(f"⏳ Cycle took {elapsed:.2f}s, sleeping {sleep_time:.2f}s until next update...")
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("🛑 Stopped by user.")

if __name__ == "__main__":
    agent = WeatherAgent(zip_file="US.txt")
    agent.run()
