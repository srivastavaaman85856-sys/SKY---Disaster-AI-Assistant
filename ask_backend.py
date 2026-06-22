import httpx
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
import os
from langdetect import detect

# Load environment variables
load_dotenv()

# Get Groq API key from .env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env file. Please add it!")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

app = FastAPI(title="Disaster Assistant Backend - Multi-Language")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Language support mapping
LANGUAGE_NAMES = {
    'en': 'English',
    'es': 'Spanish',
    'hi': 'Hindi',
    'fr': 'French',
    'de': 'German',
    'zh-cn': 'Chinese (Simplified)',
    'ar': 'Arabic',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'it': 'Italian',
    'ko': 'Korean',
    'bn': 'Bengali',
    'ur': 'Urdu',
    'tr': 'Turkish'
}


class Query(BaseModel):
    query: str
    language: str = "auto"  # Can be 'auto' or specific language code


def detect_language(text):
    """Detect the language of input text"""
    try:
        lang_code = detect(text)
        return lang_code
    except:
        return 'en'  # Default to English if detection fails


@app.post("/ask")
async def ask_ai(data: Query):
    try:
        # Detect language if set to auto
        if data.language == "auto":
            detected_lang = detect_language(data.query)
            language_name = LANGUAGE_NAMES.get(detected_lang, "English")
        else:
            detected_lang = data.language
            language_name = LANGUAGE_NAMES.get(detected_lang, "English")

        # Create system prompt with language instruction
        system_prompt = f"""You are a disaster response expert who speaks multiple languages fluently.

IMPORTANT INSTRUCTIONS:
- Respond in {language_name} language
- Provide clear, actionable advice for emergency situations
- Include specific steps and safety measures
- Be concise but thorough
- Change line of text for every single statement.
- Give point wise answer.and change line for every point.
- Use simple language that anyone can understand in emergencies
- If the user's question is in {language_name}, respond entirely in {language_name}

Focus on disasters like: earthquakes, floods, fires, hurricanes, tsunamis, tornadoes, etc."""

        # Call Groq API
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data.query}
            ],
            max_tokens=600,
            temperature=0.7
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "detected_language": detected_lang,
            "language_name": language_name
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def root():
    return {
        "status": "Backend running with Multi-Language Support 🌍",
        "supported_languages": LANGUAGE_NAMES
    }


@app.get("/languages")
def get_languages():
    """Get list of supported languages"""
    return {"languages": LANGUAGE_NAMES}

   
# ── ADD THESE ROUTES AT THE BOTTOM ──────────────────────

@app.get("/earthquakes/india")
async def get_india_earthquakes():
    params = {
        "format": "geojson",
        "minmagnitude": 4.0,
        "latitude": 20.5937,
        "longitude": 78.9629,
        "maxradiuskm": 2500,
        "orderby": "time",
        "limit": 5
    }
    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://earthquake.usgs.gov/fdsnws/event/1/query",
            params=params
        )
        data = r.json()

    result = []
    for f in data["features"]:
        p = f["properties"]
        result.append({
            "place": p["place"],
            "magnitude": p["mag"],
            "alert": p.get("alert", "green"),
            "tsunami": p["tsunami"],
            "time": p["time"]
        })
    return {"earthquakes": result}


@app.get("/offline-data")
async def offline_data():
    with open("offline_data.json") as f:
        return json.load(f)   # add 'import json' at top too