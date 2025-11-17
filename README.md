🌟 Emotion-Aware Chatbot (Voice + Text) Using AI

This project presents an Emotion-Aware Chatbot that understands human emotions from both text and voice.
It uses advanced Machine Learning, NLP, and Speech Processing to generate responses that feel more natural, empathetic, and human-like.

The goal of this project is to enhance human–AI emotional interaction, support mental wellness, and create more meaningful conversational experiences.

📌 Project Overview

The system analyzes emotions using:

📝 Text Emotion Detection

Powered by HuggingFace Transformers

Detects emotions such as:

sadness

joy

anger

fear

surprise

neutral

🎤 Voice Emotion Detection

Extracts audio features like:

MFCC

Chroma

Zero Crossing Rate

Spectral Features

Then predicts emotion using a Random Forest Classifier.

🤖 Emotion-Based Chatbot Reply

Once the emotion is detected, the chatbot generates personalized, emotionally aware responses.

🧠 Technologies & Tools Used
Machine Learning / AI

Transformers (HuggingFace)

Scikit-Learn (Random Forest)

VOSK (Offline speech-to-text)

Librosa (audio feature extraction)

Back-End

Python

Streamlit (web interface)

Data Processing

NumPy

Pandas

Soundfile

Development Tools

Git & GitHub

Virtual Environment (venv)

🎨 Web App Features

Clean & responsive Streamlit UI

Real-time emotion detection (text + voice)

Emotion-based intelligent chatbot

No internet needed for speech-to-text (VOSK)

Easy to run on any device

📁 Project Structure
emotion-aware-chatbot/
│── app.py                 # Main Streamlit app
│── requirements.txt       # Dependencies
│── README.md
│── docs/                  # Documentation & images
│── models/                # ML models
│── examples/              # Example inputs/outputs
│── src/
    │── audio_emotion.py   # Voice emotion detection
    │── text_emotion.py    # Text emotion detection
    │── chatbot.py         # Emotion-based reply generator
    │── fusion.py          # Combines audio + text emotion
    │── utils.py           # Helper functions

🚀 How to Run the Project
1. Create Virtual Environment
python -m venv venv

2. Activate (Windows)
venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Run the App
streamlit run app.py

✅ Outcome of the Project

This system can:

Understand human emotional tone

Respond empathetically

Improve human–AI interaction

Assist in emotional support applications

Help researchers build emotion-based AI systems

Can be applied in:

AI therapy chatbots

Customer support

Emotional assistants

Voice-controlled devices

📄 About

A machine-learning-powered chatbot that detects emotions from text and voice and responds with empathy.
Built using Python, Streamlit, Transformers, and ML models.