import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from gtts import gTTS
from deep_translator import GoogleTranslator 
import time
import re
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv
from geopy.geocoders import Nominatim
from streamlit_javascript import st_javascript  # ✅ For GPS detection

from model import get_model

# ====================== PAGE CONFIG ======================
st.set_page_config(page_title="AI Plant & Farming Assistant", layout="wide")

# ====================== STYLING ======================
st.markdown("""
    <style>
    h1 { color: #2e7d32; text-align: center; font-weight: 800; font-size: 2.5rem; }
    h2, h3 { color: #388e3c; font-weight: 700; }
    [data-testid="stSidebar"] { background-color: #e8f5e9; padding: 20px; border-right: 2px solid #c8e6c9; }
    div.stButton > button {
        background-color: #43a047; color: white; border: none; padding: 0.6rem 1rem;
        border-radius: 8px; font-weight: 600; transition: 0.3s;
    }
    div.stButton > button:hover { background-color: #2e7d32; transform: scale(1.05); }
    div[data-baseweb="tab-list"] > button { font-weight: bold !important; color: #1b5e20 !important; }
    img { border-radius: 10px; border: 2px solid #a5d6a7; }
    .stMarkdown p { font-size: 1.05rem; line-height: 1.6; }
    [data-testid="stMetricValue"] { color: #1b5e20; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style="text-align:center;">
        <img src="https://cdn-icons-png.flaticon.com/512/619/619032.png" width="90">
        <h1>🌿 AI Plant & Farming Assistant</h1>
        <p style="font-size:18px; color:#4e342e;">
            Smart farming with AI, weather forecasting, and multilingual support
        </p>
    </div>
""", unsafe_allow_html=True)

# ====================== ENV & MODEL ======================
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# ====================== LANGUAGES ======================
languages = {
    "English": "en", "हिन्दी": "hi", "தமிழ்": "ta", "తెలుగు": "te",
    "ಕನ್ನಡ": "kn", "മലയാളം": "ml", "मराठी": "mr", "ગુજરાતી": "gu",
    "বাংলা": "bn", "ପੰਜਾਬੀ": "pa", "ଓଡ଼ିଆ": "or"
}

# ====================== REMEDIES ======================
remedies = {
    "Apple___Apple_scab": "Apply fungicides early in the growing season. Prune infected branches and remove fallen leaves to limit spread.",
    "Apple___Black_rot": "Remove and destroy infected fruit and branches. Use fungicides during growing season and practice crop rotation.",
    "Apple___Cedar_apple_rust": "Remove nearby cedar trees if possible. Apply protective fungicide sprays on apple trees.",
    "Apple___healthy": "No remedy needed. Maintain proper care and monitoring.",
    "Blueberry___healthy": "No remedy needed. Maintain good cultivation practices.",
    "Tomato___Early_blight": "Use fungicides timely, plant resistant varieties, rotate crops.",
    "Tomato___healthy": "No remedy needed."
}

# ====================== FUNCTIONS ======================
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).resize(target_size)
    img_array = np.expand_dims(np.array(img), axis=0).astype('float32') / 255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    return class_indices[str(np.argmax(predictions, axis=1)[0])]

def translate_text(text, target_lang):
    if not text or target_lang == "en":
        return text
    try:
        return GoogleTranslator(source='en', target=target_lang).translate(text)
    except:
        return text

def clean_text_for_speech(text):
    text = re.sub(r'[*_~`]', '', text)
    return re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text).strip()

def speak_text(text, lang_code):
    try:
        tts = gTTS(clean_text_for_speech(text), lang=lang_code)
        tts.save("voice.mp3")
        st.audio("voice.mp3", format="audio/mp3")
    except Exception as e:
        st.warning(f"Voice generation failed: {e}")

def get_weather_forecast(lat, lon):
    if not OPENWEATHER_API_KEY:
        return None, "OpenWeatherMap API key not found."
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    res = requests.get(url)
    if res.status_code != 200:
        return None, f"API Error: {res.json().get('message', 'Unknown error')}"
    data = res.json()
    forecasts = []
    for e in data["list"]:
        forecasts.append({
            "date": datetime.utcfromtimestamp(e["dt"]),
            "temp": e["main"]["temp"],
            "humidity": e["main"]["humidity"],
            "rainfall": e.get("rain", {}).get("3h", 0)
        })
    df = pd.DataFrame(forecasts)
    df = df.groupby(pd.Grouper(key="date", freq="D")).mean().reset_index()
    return df, None

# ====================== TRANSLATION WRAPPER ======================
if "selected_lang" not in st.session_state:
    st.session_state["selected_lang"] = "English"
lang_code = languages[st.session_state["selected_lang"]]
def t(text): return translate_text(text, lang_code)

# ====================== LANGUAGE SELECT ======================
selected_lang = st.selectbox("🌐 Select your preferred language:", list(languages.keys()),
                             index=list(languages.keys()).index(st.session_state["selected_lang"]))
if selected_lang != st.session_state["selected_lang"]:
    st.session_state["selected_lang"] = selected_lang
    st.rerun()

# ====================== GPS LOCATION DETECTION ======================
geolocator = Nominatim(user_agent="plant_app")
st.info(t("Detecting your location automatically using GPS..."))
try:
    location_data = st_javascript("""
        async function getLocation() {
            return new Promise((resolve, reject) => {
                if (navigator.geolocation) {
                    navigator.geolocation.getCurrentPosition(
                        (pos) => resolve({lat: pos.coords.latitude, lon: pos.coords.longitude}),
                        (err) => reject(err)
                    );
                } else reject("Geolocation not supported");
            });
        }
        return await getLocation();
    """)

    if location_data and "lat" in location_data and "lon" in location_data:
        lat, lon = location_data["lat"], location_data["lon"]
        loc = geolocator.reverse((lat, lon), language="en")
        city = loc.raw['address'].get('city', loc.raw['address'].get('town', 'Unknown'))
        st.session_state["user_location"] = {"lat": lat, "lon": lon, "city": city}
        st.success(f"{t('📍 Detected Location')}: {city}")
    else:
        st.warning(t("Unable to detect GPS location. Please enable browser location access."))
except Exception as e:
    st.warning(f"{t('Could not access browser GPS.')} ({e})")

# Default fallback if no GPS
if "user_location" not in st.session_state:
    st.session_state["user_location"] = {"lat": 10.7905, "lon": 78.7047, "city": "Tiruchirappalli"}

if "ai_cache" not in st.session_state:
    st.session_state["ai_cache"] = {}

if "chat_model_tab1" not in st.session_state:
    st.session_state["chat_model_tab1"] = get_model().start_chat(history=[])

if "chat_model_tab2" not in st.session_state:
    st.session_state["chat_model_tab2"] = get_model().start_chat(history=[])

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []


# ====================== TABS ======================
st.title(t("Artificial Intelligence Powered Plant Health & Farming Assistant"))
tab1, tab2, tab3 = st.tabs([t("Disease Detection"), t("Chatbot Assistant"), t("Smart Farming Dashboard")])

# TAB 1: Plant Disease Detection
with tab1:
    st.header(t("Take or Upload a Leaf Image"))
    option = st.radio(t("Choose input method:"), (t("Use Camera"), t("Upload Image")))
    if option == t("Use Camera"):
        image_file = st.camera_input(t("Take a picture of the leaf"))
    else:
        image_file = st.file_uploader(t("Upload a leaf image..."), type=["jpg", "jpeg", "png"])
    if image_file:
        image = Image.open(image_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image.resize((200, 200)), caption=t("Captured Image"))
        with col2:
            if st.button(t("Detect Disease")):
                prediction = predict_image_class(model, image_file, class_indices)
                st.success(f"{t('Prediction')}: {prediction}")
                remedy_text = remedies.get(prediction, t("No remedy found. Consult an agronomist."))
                st.write(f"{t('Recommended Treatment')}: {remedy_text}")
                translated_pred = translate_text(prediction, lang_code)
                translated_remedy = translate_text(remedy_text, lang_code)
                st.write(f"{t('🌐 Translated')}:")
                st.write(translated_pred)
                st.write(translated_remedy)
                speak_text(remedy_text, lang_code)
                if prediction in st.session_state["ai_cache"]:
                    ai_reply = st.session_state["ai_cache"][prediction]
                else:
                    st.info(t(" Fetching AI chatbot advice..."))
                    response = st.session_state["chat_model_tab1"].send_message(
                        f"My plant has {prediction}. Suggest remedies in simple terms."
                    )
                    ai_reply = response.text.strip()
                    st.session_state["ai_cache"][prediction] = ai_reply
                    time.sleep(1)
                translated_ai_reply = translate_text(ai_reply, lang_code)
                st.write(f"{t(' Translated Advice')}:")
                st.write(translated_ai_reply)
                speak_text(translated_ai_reply, lang_code)

# TAB 2: Chatbot
with tab2:
    st.header(t("Plant Care Chatbot Assistant"))
    user_input = st.text_input(t("Ask about plants, diseases, or remedies:"))
    if st.button(t("Send")) and user_input:
        st.write(f"**{t('You')}:** {user_input}")
        st.write(f"**🤖 {t('Bot (English)')}:** {t('This is a simulated chatbot response for demo.')}")
        speak_text(t("This is a simulated chatbot response for demo."), lang_code)

# TAB 3: Smart Farming Dashboard
with tab3:
    st.header(t("Smart Farming Dashboard"))
    lat, lon, city = st.session_state["user_location"].values()
    st.success(t(f"Detected Location: {city}"))
    if st.button(t("📊 Show Forecast")):
        df, error = get_weather_forecast(lat, lon)
        if error: st.error(t(error))
        else:
            st.metric(t("Average Temperature (°C)"), f"{round(df['temp'].mean(),1)}°C")
            st.metric(t("Average Humidity (%)"), f"{round(df['humidity'].mean(),1)}%")
            st.metric(t("Average Rainfall (mm)"), f"{round(df['rainfall'].mean(),1)} mm")
            st.subheader(t("📈 Weather Trends"))
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(df["date"], df["temp"], label="Temperature (°C)", marker='o')
            ax.plot(df["date"], df["humidity"], label="Humidity (%)", marker='o')
            ax.plot(df["date"], df["rainfall"], label="Rainfall (mm)", marker='o')
            ax.legend(); ax.grid(True); plt.xticks(rotation=45)
            st.pyplot(fig)
