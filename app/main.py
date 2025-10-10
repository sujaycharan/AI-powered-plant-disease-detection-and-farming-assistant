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

# Gemini API imports
from model import get_model

# ================== LOAD ENVIRONMENT VARIABLES ==================
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# ================== MODEL LOADING ==================
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# ================== LANGUAGE SETTINGS ==================
languages = {
    "English": "en",
    "हिन्दी": "hi",
    "தமிழ்": "ta",
    "తెలుగు": "te",
    "ಕನ್ನಡ": "kn",
    "മലയാളം": "ml",
    "मराठी": "mr",
    "ગુજરાતી": "gu",
    "বাংলা": "bn",
    "ਪੰਜਾਬੀ": "pa",
    "ଓଡ଼ିଆ": "or"
}

# ================== REMEDIES ==================
remedies = {
    "Apple___Apple_scab": "Apply fungicides early in the growing season. Prune infected branches and remove fallen leaves to limit spread.",
    "Apple___Black_rot": "Remove and destroy infected fruit and branches. Use fungicides during growing season and practice crop rotation.",
    "Apple___Cedar_apple_rust": "Remove nearby cedar trees if possible. Apply protective fungicide sprays on apple trees.",
    "Apple___healthy": "No remedy needed. Maintain proper care and monitoring.",
    "Blueberry___healthy": "No remedy needed. Maintain good cultivation practices.",
    "Tomato___Early_blight": "Use fungicides timely, plant resistant varieties, rotate crops.",
    "Tomato___healthy": "No remedy needed."
}

# ================== HELPER FUNCTIONS ==================
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

def translate_text(text, target_lang):
    if not text or target_lang == "en":
        return text
    try:
        translated = GoogleTranslator(source='en', target=target_lang).translate(text)
        return translated
    except Exception as e:
        st.warning(f"⚠️ Translation failed: {e}")
        return text

def clean_text_for_speech(text):
    text = re.sub(r'[*_~`]', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    return text.strip()

def speak_text(text, lang_code):
    try:
        clean_text = clean_text_for_speech(text)
        tts = gTTS(clean_text, lang=lang_code)
        tts.save("voice.mp3")
        st.audio("voice.mp3", format="audio/mp3")
    except Exception as e:
        st.warning(f"Voice generation failed: {e}")

# ================== WEATHER DASHBOARD FUNCTIONS ==================
def get_weather_forecast(city_name):
    """Fetch 7-day forecast and extrapolate to 6 months."""
    if not OPENWEATHER_API_KEY:
        return None, "OpenWeatherMap API key not found in .env file."

    try:
        url = f"https://api.openweathermap.org/data/2.5/forecast?q={city_name}&appid={OPENWEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()

        if response.status_code != 200:
            return None, f"Error: {data.get('message', 'Unable to fetch data')}"

        forecasts = []
        for entry in data["list"]:
            date = datetime.utcfromtimestamp(entry["dt"])
            temp = entry["main"]["temp"]
            humidity = entry["main"]["humidity"]
            rainfall = entry.get("rain", {}).get("3h", 0)
            forecasts.append({"date": date, "temp": temp, "humidity": humidity, "rainfall": rainfall})

        df = pd.DataFrame(forecasts)
        df = df.groupby(df["date"].dt.date).mean().reset_index()

        # Extrapolate for 6 months (approx. 24 weeks)
        future_dates = [df["date"].iloc[-1] + timedelta(days=i * 7) for i in range(1, 25)]
        future_df = pd.DataFrame({
            "date": future_dates,
            "temp": np.clip(df["temp"].mean() + np.sin(np.linspace(0, 6, 24)) * 5, 10, 40),
            "humidity": np.clip(df["humidity"].mean() + np.cos(np.linspace(0, 6, 24)) * 10, 30, 100),
            "rainfall": np.abs(np.sin(np.linspace(0, 3, 24))) * 10
        })

        return future_df, None
    except Exception as e:
        return None, str(e)

def get_user_location():
    """Fetch user's city using IP geolocation."""
    try:
        res = requests.get("https://ipinfo.io/json")
        data = res.json()
        return data.get("city", "Chennai"), data.get("region", ""), data.get("country", "")
    except Exception:
        return "Chennai", "", ""

# ================== STREAMLIT CONFIG ==================
st.set_page_config(page_title="AI Plant & Farming Assistant", layout="wide")

if "selected_lang" not in st.session_state:
    st.session_state["selected_lang"] = "English"

selected_lang = st.selectbox(
    "Select your preferred language:",
    list(languages.keys()),
    index=list(languages.keys()).index(st.session_state["selected_lang"]),
    key="lang_select"
)

if selected_lang != st.session_state["selected_lang"]:
    st.session_state["selected_lang"] = selected_lang
    st.rerun()

lang_code = languages[st.session_state["selected_lang"]]

def t(text):
    return translate_text(text, lang_code)

if "ai_cache" not in st.session_state:
    st.session_state["ai_cache"] = {}

if "chat_model_tab1" not in st.session_state:
    st.session_state["chat_model_tab1"] = get_model().start_chat(history=[])

if "chat_model_tab2" not in st.session_state:
    st.session_state["chat_model_tab2"] = get_model().start_chat(history=[])

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ================== UI TABS ==================
st.title(t("🌿 AI-Powered Plant Health & Farming Assistant"))

tab1, tab2, tab3 = st.tabs([t("🌱 Disease Detection"), t("💬 Chatbot Assistant"), t("🌾 Smart Farming Dashboard")])

# ---------- TAB 1 ----------
with tab1:
    st.header(t("📸 Take or Upload a Leaf Image"))
    option = st.radio(t("Choose input method:"), (t("📷 Use Camera"), t("📁 Upload Image")))

    if option == t("📷 Use Camera"):
        image_file = st.camera_input(t("Take a picture of the leaf"))
    else:
        image_file = st.file_uploader(t("Upload a leaf image..."), type=["jpg", "jpeg", "png"])

    if image_file is not None:
        image = Image.open(image_file)
        col1, col2 = st.columns(2)

        with col1:
            st.image(image.resize((200, 200)), caption=t("Captured Image"))

        with col2:
            if st.button(t("🔍 Detect Disease")):
                prediction = predict_image_class(model, image_file, class_indices)
                st.success(f"{t('Prediction')}: {prediction}")

                remedy_text = remedies.get(prediction, t("No remedy found. Consult an agronomist."))
                st.write(f"{t('Recommended Treatment')}: {remedy_text}")

                translated_pred = translate_text(prediction, lang_code)
                translated_remedy = translate_text(remedy_text, lang_code)
                st.write(f"{t('🌐 Translated')}:")
                st.write(translated_pred)
                st.write(translated_remedy)
                speak_text(translated_remedy, lang_code)

                if prediction in st.session_state["ai_cache"]:
                    ai_reply = st.session_state["ai_cache"][prediction]
                else:
                    st.info(t("🤖 Fetching AI chatbot advice..."))
                    response = st.session_state["chat_model_tab1"].send_message(
                        f"My plant has {prediction}. Suggest remedies in simple terms."
                    )
                    ai_reply = response.text.strip()
                    st.session_state["ai_cache"][prediction] = ai_reply
                    time.sleep(1)

                translated_ai_reply = translate_text(ai_reply, lang_code)
                st.write(f"{t('🌐 Translated Advice')}:")
                st.write(translated_ai_reply)
                speak_text(translated_ai_reply, lang_code)

# ---------- TAB 2 ----------
with tab2:
    st.header(t("💬 Plant Care Chatbot Assistant"))
    user_input = st.text_input(t("Ask me anything about plants, diseases, or remedies:"))

    if st.button(t("Send")):
        if user_input:
            st.session_state["chat_history"].append((t("You"), user_input))
            if user_input in st.session_state["ai_cache"]:
                ai_reply = st.session_state["ai_cache"][user_input]
            else:
                response = st.session_state["chat_model_tab2"].send_message(user_input)
                ai_reply = response.text.strip()
                st.session_state["ai_cache"][user_input] = ai_reply
                time.sleep(1)

            st.write(f"{t('🤖 Bot (English)')}:")
            st.write(ai_reply)

            translated_reply = translate_text(ai_reply, lang_code)
            st.write(f"{t('🌐 Translated')}:")
            st.write(translated_reply)
            speak_text(translated_reply, lang_code)

            st.session_state["chat_history"].append((t("Bot"), ai_reply))

    st.subheader(t("🧾 Chat History"))
    for role, text in st.session_state["chat_history"]:
        st.markdown(f"**{role}:** {text}")

# ---------- TAB 3: SMART FARMING DASHBOARD ----------
with tab3:
    st.header(t("🌾 Smart Farming Dashboard"))
    st.write(t("Get 6-month weather insights automatically based on your location."))

    with st.spinner(t("Detecting your location...")):
        city, region, country = get_user_location()

    st.success(t(f"📍 Detected Location: {city}, {region}, {country}"))

    if st.button(t("📊 Show Forecast")):
        df, error = get_weather_forecast(city)
        if error:
            st.error(t(error))
        else:
            st.success(t(f"Weather forecast for {city} (Next 6 Months)"))

            avg_temp = round(df["temp"].mean(), 1)
            avg_humidity = round(df["humidity"].mean(), 1)
            avg_rainfall = round(df["rainfall"].mean(), 1)

            st.metric(t("🌡️ Avg Temperature (°C)"), f"{avg_temp}°C")
            st.metric(t("💧 Avg Humidity (%)"), f"{avg_humidity}%")
            st.metric(t("🌧️ Avg Rainfall (mm)"), f"{avg_rainfall} mm")

            st.subheader(t("📈 6-Month Climate Prediction Trends"))
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(df["date"], df["temp"], label=t("Temperature (°C)"), marker='o')
            ax.plot(df["date"], df["humidity"], label=t("Humidity (%)"), marker='o')
            ax.plot(df["date"], df["rainfall"], label=t("Rainfall (mm)"), marker='o')
            ax.set_xlabel(t("Month"))
            ax.set_ylabel(t("Values"))
            ax.legend()
            ax.grid(True)
            plt.xticks(rotation=45)
            st.pyplot(fig)
