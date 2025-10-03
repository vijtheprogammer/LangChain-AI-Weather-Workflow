import os
import csv
import asyncio
import aiohttp
import pandas as pd
import requests
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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

# 2. Async weather fetch
async def fetch_weather(session, zip_code):
    url = f"http://api.openweathermap.org/data/2.5/weather?zip={zip_code},us&appid={WEATHER_API_KEY}&units=metric"
    try:
        async with session.get(url, timeout=10) as response:
            if response.status != 200:
                print(f"❌ Failed {zip_code} - Status {response.status}")
                return None
            data = await response.json()
            return {
                "Zip Code": zip_code,
                "Description": data["weather"][0]["description"],
                "Temperature (°C)": data["main"]["temp"],
                "Min Temp (°C)": data["main"]["temp_min"],
                "Max Temp (°C)": data["main"]["temp_max"]
            }
    except Exception as e:
        print(f"⚠️ Error fetching {zip_code}: {e}")
        return None

async def get_all_weather(zip_codes, concurrency_limit=20):
    connector = aiohttp.TCPConnector(limit_per_host=concurrency_limit)
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [fetch_weather(session, zip_code) for zip_code in zip_codes]
        results = await asyncio.gather(*tasks)
    return [r for r in results if r]

# 3. Save data to CSV (ORIGINAL)
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

# 4. AI Summary using LangChain (ORIGINAL)
def generate_summary_with_llm(weather_df: pd.DataFrame) -> str:
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

# 5. Send summary and CSV to Discord (ORIGINAL)
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

# 6. NEW: Calculate average weather stats
def calculate_average_weather(weather_data):
    """Calculate average weather metrics across all ZIP codes"""
    if not weather_data:
        return None
    
    df = pd.DataFrame(weather_data)
    
    avg_stats = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "avg_temp": round(df["Temperature (°C)"].mean(), 2),
        "avg_min_temp": round(df["Min Temp (°C)"].mean(), 2),
        "avg_max_temp": round(df["Max Temp (°C)"].mean(), 2),
        "num_locations": len(weather_data),
        "most_common_condition": df["Description"].mode()[0] if not df["Description"].mode().empty else "N/A"
    }
    
    return avg_stats

# 7. NEW: Log trends to CSV
def log_weather_trends(avg_stats, trends_file="weather_trends.csv"):
    """Append average weather stats to trends log"""
    if not avg_stats:
        print("No stats to log.")
        return
    
    file_exists = os.path.isfile(trends_file)
    
    with open(trends_file, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=avg_stats.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(avg_stats)
    
    print(f"📊 Logged trends to {trends_file}")

# 8. NEW: AI Brain - Decision Making
def ai_brain_analyze(current_data, previous_data=None):
    """AI analyzes weather data and decides what to do next"""
    
    print("\n" + "="*60)
    print("🧠 AI BRAIN THINKING...")
    print("="*60)
    
    if not current_data:
        print("❌ No data to analyze")
        return {"action": "wait", "interval": 60, "reason": "No data available"}
    
    current_df = pd.DataFrame(current_data)
    current_avg = current_df["Temperature (°C)"].mean()
    
    # Build context for AI
    context = f"""You are an intelligent weather monitoring agent. Analyze the current weather data and decide what to do next.

Current Weather Stats:
- Average Temperature: {round(current_avg, 2)}°C
- Number of locations: {len(current_data)}
- Temperature range: {round(current_df['Temperature (°C)'].min(), 2)}°C to {round(current_df['Temperature (°C)'].max(), 2)}°C
- Most common condition: {current_df['Description'].mode()[0] if not current_df['Description'].mode().empty else 'N/A'}
"""
    
    if previous_data:
        prev_df = pd.DataFrame(previous_data)
        prev_avg = prev_df["Temperature (°C)"].mean()
        temp_change = current_avg - prev_avg
        
        context += f"""
Previous Cycle Comparison:
- Previous average temperature: {round(prev_avg, 2)}°C
- Temperature change: {round(temp_change, 2)}°C
- Change rate: {round(abs(temp_change), 2)}°C per minute
"""
    
    context += """
Based on this data, decide:
1. What is the situation? (normal, interesting, urgent)
2. What action should I take? (wait, alert, investigate)
3. How long until next check? (60 seconds for normal, 30 for interesting, 15 for urgent)
4. Why did you make this decision?

IMPORTANT: Be more sensitive to changes. Consider these as noteworthy:
- ANY temperature change > 0.5°C
- Any new weather condition appearing
- Temperature variance across locations
- Even subtle patterns worth monitoring

Err on the side of "interesting" or "alert" rather than "wait". We want frequent updates.

Respond in JSON format:
{
    "situation": "normal|interesting|urgent",
    "action": "wait|alert|investigate",
    "interval": 60,
    "reason": "brief explanation of your thinking",
    "notable_observations": ["observation1", "observation2"]
}
"""
    
    # Call AI
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    result = llm.invoke(context)
    
    # Parse response
    import json
    import re
    
    # Extract JSON from response (handle markdown code blocks)
    content = result.content.strip()
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        decision = json.loads(json_match.group())
    else:
        decision = {
            "situation": "normal",
            "action": "wait", 
            "interval": 60,
            "reason": "Unable to parse AI response",
            "notable_observations": []
        }
    
    # Display AI's thinking
    print(f"\n💭 SITUATION: {decision['situation'].upper()}")
    print(f"🎯 ACTION: {decision['action'].upper()}")
    print(f"⏱️  NEXT CHECK: {decision['interval']} seconds")
    print(f"\n📝 REASONING:")
    print(f"   {decision['reason']}")
    
    if decision.get('notable_observations'):
        print(f"\n👀 OBSERVATIONS:")
        for obs in decision['notable_observations']:
            print(f"   • {obs}")
    
    print("="*60 + "\n")
    
    return decision

# 9. NEW: Generate scatter plot from trends
def generate_trend_plot(trends_file="weather_trends.csv", plot_file="weather_trends_plot.png"):
    """Create a scatter plot showing temperature trends over time"""
    
    if not os.path.isfile(trends_file):
        print("No trends file found yet.")
        return
    
    # Read trends data
    df = pd.read_csv(trends_file)
    
    if len(df) < 2:
        print("Not enough data points for plotting yet.")
        return
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Scatter plots for different temperature metrics
    plt.scatter(df['timestamp'], df['avg_temp'], label='Average Temp', alpha=0.7, s=50, color='blue')
    plt.scatter(df['timestamp'], df['avg_min_temp'], label='Average Min', alpha=0.5, s=30, color='green')
    plt.scatter(df['timestamp'], df['avg_max_temp'], label='Average Max', alpha=0.5, s=30, color='red')
    
    # Formatting
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.title('Weather Temperature Trends Over Time', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    plt.gcf().autofmt_xdate()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    
    print(f"📈 Saved trend plot to {plot_file}")

# 10. NEW: Send plot to Discord
def send_plot_to_discord(plot_file="weather_trends_plot.png"):
    """Send the trend plot to Discord"""
    if not DISCORD_WEBHOOK_URL:
        print("DISCORD_WEBHOOK_URL not set.")
        return
    
    if not os.path.isfile(plot_file):
        print("Plot file not found.")
        return
    
    try:
        with open(plot_file, 'rb') as file:
            payload = {"content": "📈 **Temperature Trends Visualization**"}
            files = {"file": (os.path.basename(plot_file), file)}
            response = requests.post(DISCORD_WEBHOOK_URL, data=payload, files=files)
            response.raise_for_status()
            print("📤 Plot sent to Discord.")
    except Exception as e:
        print(f"Failed to send plot to Discord: {e}")

# 11. NEW: Main automated loop with AI Brain
def run_automated_monitoring():
    """Run weather monitoring with AI decision-making"""
    
    zip_codes = load_zip_codes_from_csv("Zip_Codes.csv")
    print(f"🚀 Starting AI-powered weather monitoring for {len(zip_codes)} ZIP codes")
    print(f"⏱️  Initial check interval: 60 seconds (1 minute)")
    print(f"🧠 AI Brain will adapt behavior based on observations")
    print(f"   Press Ctrl+C to stop.\n")
    
    cycle_count = 0
    previous_weather_data = None
    next_check_interval = 60  # Start with 1 minute
    
    try:
        while True:
            cycle_count += 1
            print(f"\n{'='*50}")
            print(f"🔄 Cycle #{cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*50}")
            
            # Fetch weather data
            weather_data = asyncio.run(get_all_weather(zip_codes, concurrency_limit=20))
            print(f"✅ Collected weather for {len(weather_data)} locations.")
            
            # AI BRAIN ANALYZES AND DECIDES
            decision = ai_brain_analyze(weather_data, previous_weather_data)
            
            # Save to CSV (ORIGINAL)
            output_csv = "weather_output.csv"
            save_weather_data_to_csv(weather_data, output_csv)
            
            # AI summary (ORIGINAL) - only if AI decides to alert
            if decision['action'] in ['alert', 'investigate']:
                df = pd.DataFrame(weather_data)
                summary = generate_summary_with_llm(df)
                print(f"\n🤖 AI Summary (triggered by {decision['action']}):\n{summary}\n")
                
                # Send to Discord (ORIGINAL)
                send_results_to_discord(summary, output_csv)
            else:
                print(f"ℹ️  AI decided not to generate full summary (situation: {decision['situation']})")
            
            # NEW: Calculate and log trends
            avg_stats = calculate_average_weather(weather_data)
            log_weather_trends(avg_stats)
            
            # NEW: Generate and send plot every cycle
            generate_trend_plot()
            if decision['action'] == 'alert':
                send_plot_to_discord()
            
            # Store for next comparison
            previous_weather_data = weather_data
            
            # AI determines wait time
            next_check_interval = decision.get('interval', 60)
            
            # Wait for next check
            print(f"\n⏳ AI Brain decided to wait {next_check_interval} seconds until next check...")
            time.sleep(next_check_interval)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped by user.")
        print(f"📊 Completed {cycle_count} cycles.")
        print(f"📁 Check 'weather_trends.csv' and 'weather_trends_plot.png' for analysis.")

if __name__ == "__main__":
    run_automated_monitoring()
