# Weather Data Analyzer

## Overview

A Python automation tool that fetches weather data for multiple U.S. ZIP codes, generates AI-powered summaries using OpenAI's GPT-4, and delivers results directly to Discord. Perfect for monitoring weather patterns across large geographical areas.

## Features

- **Batch Weather Data Collection**: Fetch current weather for hundreds of ZIP codes automatically
- **Rate Limit Management**: Built-in delays to respect API rate limits
- **AI-Powered Summaries**: Uses LangChain and GPT-4 to generate insightful weather pattern analysis
- **Discord Integration**: Automatically sends summaries and CSV reports to your Discord channel
- **CSV Export**: Saves all weather data in a structured CSV format for further analysis

## Prerequisites

- Python 3.8 or higher
- OpenWeather API key (free tier available)
- Discord webhook URL
- OpenAI API key

## Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/weather-data-analyzer.git
cd weather-data-analyzer
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Create Requirements File

Create a `requirements.txt` file with the following dependencies:

```
requests
pandas
python-dotenv
langchain-openai
langchain
openai
```

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
WEATHER_API_KEY=your_openweather_api_key_here
DISCORD_WEBHOOK_URL=your_discord_webhook_url_here
OPENAI_API_KEY=your_openai_api_key_here
```

### API Keys Setup

#### OpenWeather API Key

1. Visit [OpenWeatherMap](https://openweathermap.org/api)
2. Sign up for a free account
3. Generate an API key from your dashboard
4. Add it to your `.env` file

#### Discord Webhook URL

1. Open your Discord server
2. Go to Server Settings → Integrations → Webhooks
3. Click "New Webhook" or edit an existing one
4. Copy the webhook URL
5. Add it to your `.env` file

#### OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/)
2. Create an account and add billing information
3. Generate an API key from the API keys section
4. Add it to your `.env` file

## Input Data Format

### ZIP Codes CSV

Create a `Zip_Codes.csv` file with one of the following formats:

**Option 1:**
```csv
ZIPCODE
10001
10002
10003
```

**Option 2:**
```csv
ZIP_CODE_TEXT
10001
10002
10003
```

The script automatically handles both column name formats.

## Usage

### Run the Script

```bash
python weather_analyzer.py
```

### What Happens

1. **Loads ZIP codes** from `Zip_Codes.csv`
2. **Fetches weather data** for each ZIP code in batches of 50
3. **Waits 60 seconds** between batches to respect API rate limits
4. **Saves results** to `weather_output.csv`
5. **Generates AI summary** using GPT-4
6. **Sends everything** to your Discord channel

## Output

### CSV File Structure

The `weather_output.csv` contains:

- **Zip Code**: The ZIP code queried
- **Description**: Weather condition description
- **Temperature (°C)**: Current temperature
- **Min Temp (°C)**: Minimum temperature
- **Max Temp (°C)**: Maximum temperature

### Discord Message

Includes:
- AI-generated weather summary with key insights
- Attached CSV file with complete data

## Configuration Options

### Batch Size and Delay

Modify these variables in the `main()` function:

```python
chunk_size = 50        # Number of ZIP codes per batch
delay_seconds = 60     # Wait time between batches
```

### AI Summary Customization

Adjust the LLM settings in `generate_summary_with_llm()`:

```python
llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
```

- **model**: Change to "gpt-3.5-turbo" for faster/cheaper results
- **temperature**: Adjust between 0.0 (deterministic) and 1.0 (creative)

## Error Handling

The script includes robust error handling:

- **Network Errors**: Timeouts and connection issues are caught and logged
- **Invalid ZIP Codes**: Skipped with error messages
- **API Rate Limits**: Managed through batch processing with delays
- **Missing Environment Variables**: Gracefully handled with informative messages

## Rate Limiting

### OpenWeather API

- Free tier: 60 calls/minute, 1,000,000 calls/month
- Script processes 50 ZIP codes per minute by default
- Adjust `chunk_size` and `delay_seconds` based on your plan

### OpenAI API

- Rate limits vary by account tier
- Script makes one API call for the entire dataset
- Monitor your usage on the OpenAI dashboard

## Troubleshooting

### "No data to save" Error

- Check your `Zip_Codes.csv` file format
- Ensure ZIP codes are valid U.S. ZIP codes
- Verify your OpenWeather API key is active

### Discord Webhook Not Working

- Verify the webhook URL is correct
- Check the webhook hasn't been deleted in Discord
- Ensure your bot has permission to post in the channel

### API Key Errors

- Confirm all API keys are in your `.env` file
- Check for extra spaces or quotes in your `.env` file
- Verify your API keys are active and have available credits

## Acknowledgments

- [OpenWeatherMap](https://openweathermap.org/) for weather data API
- [LangChain](https://www.langchain.com/) for LLM integration framework
- [OpenAI](https://openai.com/) for GPT-4 language model

## Contact

For questions or support, please open an issue on GitHub.

## Roadmap

- [ ] Add support for international ZIP/postal codes
- [ ] Implement weather alerts and warnings
- [ ] Add historical weather data comparison
- [ ] Create interactive visualization dashboard
- [ ] Support for multiple Discord channels
- [ ] Add email notification option
