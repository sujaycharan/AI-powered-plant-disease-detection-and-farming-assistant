# 🌿 AI-Powered Plant Health & Smart Farming Assistant

## 🧠 Project Overview
This project is an **AI-driven plant health assistant** that helps farmers and plant enthusiasts detect plant diseases, get localized weather forecasts, and chat with an AI-powered farming assistant — all in their **preferred regional language**.

Built using **Streamlit**, **TensorFlow**, and **OpenWeatherMap API**, this web app provides a complete smart farming solution powered by artificial intelligence.

---

## 🚀 Key Features

### 🌱 1. AI Plant Disease Detection
- Upload or capture a **leaf image** directly using your camera.
- Uses a **deep learning CNN model** trained on plant disease datasets.
- Displays **predicted disease name** and **recommended remedies**.
- Supports **multilingual translation** of results using Google Translator.
- Includes **voice narration** for remedies via Google Text-to-Speech (gTTS).

---

### 💬 2. Plant Care Chatbot Assistant
- Integrated with **Google Gemini AI** chatbot for intelligent, conversational help.
- Ask anything about **plant diseases, fertilizers, remedies, or care tips**.
- Automatically translates both **user input** and **AI responses** into your selected language.
- Saves **chat history** for better continuity.
- Uses caching to avoid redundant translation or AI calls for repeated queries.

---

### 🌾 3. Smart Weather Forecast Dashboard
- Fetches **live and forecasted weather** data using the **OpenWeatherMap API**.
- Uses **browser GPS** to auto-detect user’s exact location via reverse geocoding.
- Users can manually enter a **city name** if GPS access fails.
- Displays **6-month extended forecast** (temperature, humidity, rainfall trends).
- Visualizes weather insights using clean **line charts** (Matplotlib + Streamlit).

---

## 🌐 Multilingual Support
Supports the following Indian languages:

| Language | Code |
|-----------|------|
| English | en |
| हिन्दी | hi |
| தமிழ் | ta |
| తెలుగు | te |
| ಕನ್ನಡ | kn |
| മലയാളം | ml |
| मराठी | mr |
| ગુજરાતી | gu |
| বাংলা | bn |
| ਪੰਜਾਬੀ | pa |
| ଓଡ଼ିଆ | or |

Every interface text, chatbot reply, and remedy can be dynamically translated without reloading the app.

---

## ⚙️ Tech Stack

| Component | Technology Used |
|------------|-----------------|
| Frontend | Streamlit |
| Backend | Python |
| AI Model | TensorFlow (CNN Model) |
| Translation | Deep Translator (Google Translator API) |
| Voice Output | gTTS (Google Text-to-Speech) |
| Weather | OpenWeatherMap API |
| Geolocation | HTML5 Geolocation + Nominatim Reverse Geocoding |
| Chatbot | Google Gemini API |
| Data Handling | NumPy, Pandas, Matplotlib |

---

## 🧩 Project Structure
```
📂 plant-disease-prediction-cnn-deep-learning-project
│
├── 📁 app
│   ├── main.py                # Streamlit main app
│   ├── model.py               # Gemini AI integration
│   ├── trained_model/
│   │   └── plant_disease_prediction_model.h5
│   ├── class_indices.json     # Mapping for class indices
│   └── .env                   # API keys (OpenWeatherMap, Gemini)
│
├── requirements.txt
└── README.md
```

---

## 🔑 Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/plant-disease-ai-assistant.git
cd plant-disease-ai-assistant/app
```

### 2️⃣ Create and Activate Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate     # Windows
# or
source venv/bin/activate  # macOS/Linux
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Set Up `.env` File
Create a file named `.env` inside the `app/` folder and add your API keys:

```env
OPENWEATHER_API_KEY=your_openweather_api_key
GOOGLE_API_KEY=your_gemini_api_key
```

### 5️⃣ Run the Application
```bash
streamlit run main.py
```

---

## 🌍 Browser Permissions
Make sure to **allow location access** in your browser when prompted.  
If GPS detection fails, you can manually enter your **city name**, and the dashboard will update automatically.

---

## 🧠 How It Works
1. The user uploads or captures a plant leaf image.
2. The CNN model predicts the disease and fetches remedies.
3. The app translates and narrates the remedy in the selected language.
4. The chatbot (Gemini AI) provides detailed explanations or advice.
5. The weather dashboard auto-detects location and shows forecasts.

---

## 💡 Future Enhancements
- Real-time pest and soil health detection using IoT sensors.
- Integration with government agricultural APIs for crop alerts.
- SMS notifications for rural farmers without internet access.
- Cloud-based model deployment for faster inference.

---

## 👨‍💻 Developed By
**Sujay Charan**  
🎓 Computer Science and Engineering Student  
💬 Passionate about AI, ML, and Smart Agriculture Solutions

---

## 🏆 Acknowledgments
- TensorFlow for deep learning support  
- Streamlit for interactive UI  
- OpenWeatherMap for weather API  
- Google Gemini AI for chatbot intelligence  
- Deep Translator for multilingual support  
- gTTS for text-to-speech capabilities

---

