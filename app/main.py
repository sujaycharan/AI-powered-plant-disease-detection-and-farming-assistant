import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
from gtts import gTTS
from deep_translator import GoogleTranslator
import time

# Gemini API imports
from model import get_model

# ================== MODEL LOADING ==================
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model.h5"
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# ================== LANGUAGE SETTINGS ==================
languages = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Bengali": "bn",
    "Punjabi": "pa",
    "Odia": "or"
}

# ================== UI TEXTS ==================
ui_texts = {
    "app_title": "🌿 AI-Powered Plant Health Assistant",
    "language_select": "Select your preferred language:",
    "tab1": "🌱 Disease Detection",
    "tab2": "💬 Chatbot Assistant",
    "header_tab1": "📸 Take or Upload a Leaf Image",
    "input_method": "Choose input method:",
    "camera_option": "📷 Use Camera",
    "upload_option": "📁 Upload Image",
    "detect_button": "🔍 Detect Disease",
    "recommended_treatment": "**Recommended Treatment:**",
    "chat_input": "Ask me anything about plants, diseases, or remedies:",
    "send_button": "Send",
    "chat_history": "🧾 Chat History",
    "ai_fetching": "🤖 Fetching AI chatbot advice..."
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

def speak_text(text, lang_code):
    try:
        tts = gTTS(text, lang=lang_code)
        tts.save("voice.mp3")
        st.audio("voice.mp3", format="audio/mp3")
    except Exception as e:
        st.warning(f"Voice generation failed: {e}")

def translate_ui(key, lang_code):
    text = ui_texts.get(key, key)
    return translate_text(text, lang_code)

# ================== STREAMLIT UI ==================
st.set_page_config(page_title="Plant Disease & Chatbot", layout="wide")

# ----------------- LANGUAGE SELECTION -----------------
if "selected_lang" not in st.session_state:
    st.session_state["selected_lang"] = "English"

def change_language():
    st.session_state["selected_lang"] = st.session_state["lang_select"]
    st.rerun()

st.selectbox(
    translate_ui("language_select", "en"),
    list(languages.keys()),
    index=list(languages.keys()).index(st.session_state["selected_lang"]),
    key="lang_select",
    on_change=change_language
)

lang_code = languages[st.session_state["selected_lang"]]
st.title(translate_ui("app_title", lang_code))

# ================== SESSION STATE INITIALIZATION ==================
if "ai_cache" not in st.session_state:
    st.session_state["ai_cache"] = {}

if "chat_model_tab1" not in st.session_state:
    st.session_state["chat_model_tab1"] = get_model().start_chat(history=[])

if "chat_model_tab2" not in st.session_state:
    st.session_state["chat_model_tab2"] = get_model().start_chat(history=[])

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# ----------------- TABS -----------------
tab1, tab2 = st.tabs([translate_ui("tab1", lang_code), translate_ui("tab2", lang_code)])

# ---------- TAB 1: PLANT DISEASE DETECTION ----------
with tab1:
    st.header(translate_ui("header_tab1", lang_code))
    option = st.radio(
        translate_ui("input_method", lang_code),
        (translate_ui("camera_option", lang_code), translate_ui("upload_option", lang_code))
    )

    if option == translate_ui("camera_option", lang_code):
        image_file = st.camera_input(translate_ui("header_tab1", lang_code))
    else:
        image_file = st.file_uploader(translate_ui("header_tab1", lang_code), type=["jpg", "jpeg", "png"])

    if image_file is not None:
        image = Image.open(image_file)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image.resize((200, 200)), caption=translate_ui("header_tab1", lang_code))
        with col2:
            if st.button(translate_ui("detect_button", lang_code)):
                prediction = predict_image_class(model, image_file, class_indices)

                remedy_text = remedies.get(prediction, translate_ui("recommended_treatment", lang_code))
                st.write(f"{translate_ui('recommended_treatment', lang_code)} {translate_text(remedy_text, lang_code)}")

                # Translate prediction
                translated_pred = translate_text(remedy_text, lang_code)
                st.write(f"🌐 {translated_pred}")
                speak_text(translated_pred, lang_code)

                # ------------------ AI Chatbot Advice with caching ------------------
                if prediction in st.session_state["ai_cache"]:
                    ai_reply = st.session_state["ai_cache"][prediction]
                else:
                    st.info(translate_ui("ai_fetching", lang_code))
                    response = st.session_state["chat_model_tab1"].send_message(
                        f"My plant has {prediction}. Suggest remedies in simple terms."
                    )
                    ai_reply = response.text.strip()
                    st.session_state["ai_cache"][prediction] = ai_reply
                    time.sleep(1)

                translated_ai_reply = translate_text(ai_reply, lang_code)
                st.write(f"🌐 {translated_ai_reply}")
                speak_text(translated_ai_reply, lang_code)

# ---------- TAB 2: CHATBOT ASSISTANT ----------
with tab2:
    st.header(translate_ui("tab2", lang_code))
    user_input = st.text_input(translate_ui("chat_input", lang_code))
    if st.button(translate_ui("send_button", lang_code)):
        if user_input:
            st.session_state["chat_history"].append(("You", user_input))

            if user_input in st.session_state["ai_cache"]:
                ai_reply = st.session_state["ai_cache"][user_input]
            else:
                response = st.session_state["chat_model_tab2"].send_message(user_input)
                ai_reply = response.text.strip()
                st.session_state["ai_cache"][user_input] = ai_reply
                time.sleep(1)

            st.write(f"**🤖 Bot:** {ai_reply}")
            translated_reply = translate_text(ai_reply, lang_code)
            st.write(f"🌐 {translated_reply}")
            speak_text(translated_reply, lang_code)
            st.session_state["chat_history"].append(("Bot", ai_reply))

    st.subheader(translate_ui("chat_history", lang_code))
    for role, text in st.session_state["chat_history"]:
        st.markdown(f"**{role}:** {text}")
