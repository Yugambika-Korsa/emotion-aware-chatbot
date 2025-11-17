# app.py (paste-replace your existing file)
import streamlit as st
import os
import librosa
import wave, json, io, csv, datetime
from pathlib import Path

# local modules (must exist in src/)
from src.utils import record, load_audio
from src.audio_emotion import extract_features, AudioEmotionModel, prepare_audio_file
from src.text_emotion import analyze_text
from src.fusion import fuse
from src.chatbot import generate_reply

# --- Page config ---
st.set_page_config(page_title="Emotion-Aware Chatbot", layout="wide", initial_sidebar_state="expanded")

# --- Styles (bubbles, colors) ---
CSS = """
<style>
:root{
  --bg:#0f1720;
  --card:#0b1220;
  --muted:#9aa6b2;
}
body { background-color: white; }
.header {
  display:flex; align-items:center; gap:12px;
}
.title { font-size:24px; font-weight:700; margin:0; }
.subtitle { color:#6b7280; margin:0; font-size:13px; }

.chat-container { padding: 12px; }
.msg-row { display:flex; gap:10px; margin:8px 0; align-items:flex-end; }
.msg-bubble {
  max-width:70%;
  padding:10px 14px;
  border-radius:14px;
  line-height:1.3;
  box-shadow: 0 1px 6px rgba(16,24,40,0.06);
}
.msg-you { background:#E6F0FF; color:#0B3A8C; align-self:flex-end; border-bottom-right: 2px solid #cfe3ff; }
.msg-bot { background:#F3F4F6; color:#0B1220; align-self:flex-start; border-bottom-left: 2px solid #e2e8f0; }
.meta { font-size:11px; color:#6b7280; margin-top:6px; }
.badge { display:inline-block; padding:4px 8px; border-radius:999px; font-size:12px; margin-left:6px; }
.badge-happy { background: #FEF3C7; color:#7C4D00; }
.badge-sad { background:#E0F2FE; color:#024873; }
.badge-anger { background:#FFE4E6; color:#6B0219; }
.badge-neutral { background:#EDE9FE; color:#2B0B5A; }
.small { font-size:12px; color:#6b7280; }
.controls { display:flex; gap:8px; align-items:center; }
.card {
  padding: 16px;
  border-radius: 10px;
  background: white;
  box-shadow: 0 6px 18px rgba(11, 22, 40, 0.04);
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --- Helper UI functions ---
def emotion_badge_html(label: str, confidence: float):
    label = (label or "neutral").lower()
    cls = "badge-neutral"
    if "joy" in label or "happy" in label:
        cls = "badge-happy"
    elif "sad" in label:
        cls = "badge-sad"
    elif "angry" in label or "anger" in label:
        cls = "badge-anger"
    return f"<span class='badge {cls}'>{label.title()} · {confidence:.2f}</span>"

def render_chat_history(history):
    """history: list of dicts {who:'user'|'bot', text:str, emotion:str, confidence:float, ts:iso}"""
    for item in history:
        who = item.get("who")
        text = item.get("text", "")
        emotion = item.get("emotion", "neutral")
        conf = float(item.get("confidence", 0.0))
        ts = item.get("ts", "")
        if who == "user":
            st.markdown(f"<div class='msg-row' style='justify-content:flex-end'><div class='msg-bubble msg-you'>{st.markdown(text, unsafe_allow_html=False)}<div class='meta small'>You • {ts}</div></div></div>", unsafe_allow_html=True)
        else:
            # bot bubble with badge inline
            badge = emotion_badge_html(emotion, conf)
            # We'll render bot bubble manually and include badge and timestamp under it
            html = f"""
            <div class='msg-row' style='justify-content:flex-start'>
              <div class='msg-bubble msg-bot'>
                {text}
                <div class='meta small'>{badge} • {ts}</div>
              </div>
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

# --- Sidebar settings ---
st.sidebar.markdown("## Settings")
CONF_THRESHOLD = st.sidebar.slider("Confidence threshold", min_value=0.0, max_value=0.9, value=0.40, step=0.05,
                                   help="If model confidence is below this, fallback to neutral.")
show_conf = st.sidebar.checkbox("Show confidence in chat", value=False)
if st.sidebar.button("Clear conversation"):
    st.session_state.chat_history = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "audio_model" not in st.session_state:
    # load model lazily
    st.session_state.audio_model = AudioEmotionModel()

# --- Header ---
colh1, colh2 = st.columns([9,3])
with colh1:
    st.markdown("<div class='header'><div><h1 class='title'>Emotion-Aware Chatbot</h1><div class='subtitle'>Converts your voice & text into empathetic replies — local and private.</div></div></div>", unsafe_allow_html=True)

with colh2:
    st.markdown("<div style='text-align:right'><small class='small'>v1.0 • Local</small></div>", unsafe_allow_html=True)

st.write("")  # spacing

# --- Main layout ---
left, right = st.columns([3,6])

# LEFT: Voice controls and audio emotion summary
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Voice (record)")
    duration = st.slider("Record length (sec)", 1, 8, 4)
    record_col1, record_col2 = st.columns([1,3])
    with record_col1:
        if st.button("🔴 Record"):
            wav_path = record(duration=duration, sr=16000)
            st.session_state._last_wav = wav_path
    with record_col2:
        if " _last_wav" in st.session_state:
            pass
    # Play last recording if any
    if st.session_state.get("_last_wav"):
        wav_path = st.session_state["_last_wav"]
        st.audio(wav_path)
        st.write("Saved:", wav_path)
        # process audio for emotion
        try:
            y, sr = prepare_audio_file(wav_path, sr=22050)
            feats = extract_features(y, sr)
            try:
                audio_model = st.session_state.audio_model
                audio_label = audio_model.predict(feats)
                try:
                    probs = audio_model.predict_proba(feats)
                    audio_scores = {str(l).lower(): float(p) for l, p in zip(audio_model.model.classes_, probs)}
                except Exception:
                    audio_scores = {str(audio_label).lower(): 1.0}
            except Exception:
                audio_label = "neutral"
                audio_scores = {"neutral": 1.0}
            st.markdown(f"**Audio emotion:** {audio_label} · confidence {max(audio_scores.values()):.2f}")
        except Exception as e:
            st.info("Audio processing error or no audio model present. Using neutral audio.")
            audio_label, audio_scores = "neutral", {"neutral": 1.0}
    else:
        st.info("Record voice and the app will analyze audio emotion here.")

    st.markdown("</div>", unsafe_allow_html=True)

# RIGHT: Chat interface
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Chat")
    # chat history display
    if st.session_state.chat_history:
        for msg in st.session_state.chat_history[-20:]:
            who = msg["who"]
            text = msg["text"]
            emotion = msg.get("emotion", "neutral")
            conf = float(msg.get("confidence", 0.0))
            ts = msg.get("ts", "")
            # render bubbles with controlled html:
            if who == "You":
                bubble_html = f"""
                <div class='msg-row' style='justify-content:flex-end'>
                  <div class='msg-bubble msg-you'>{text}
                    <div class='meta small'>You • {ts}</div>
                  </div>
                </div>
                """
                st.markdown(bubble_html, unsafe_allow_html=True)
            else:
                badge = emotion_badge_html(emotion, conf)
                bubble_html = f"""
                <div class='msg-row' style='justify-content:flex-start'>
                  <div class='msg-bubble msg-bot'>{text}
                    <div class='meta small'>{badge} • {ts}</div>
                  </div>
                </div>
                """
                st.markdown(bubble_html, unsafe_allow_html=True)

    st.write("")  # spacer

    # Input area
    input_col, send_col = st.columns([8,2])
    with input_col:
        user_text = st.text_area("Type your message here...", key="input_text", height=100)
    with send_col:
        if st.button("Send"):
            if not user_text.strip():
                st.info("Please type something first.")
            else:
                # Analyze text emotion
                try:
                    txt_label, txt_scores = analyze_text(user_text)
                    # normalize type
                    if not isinstance(txt_scores, dict):
                        try:
                            txt_scores = dict(txt_scores)
                        except Exception:
                            txt_scores = {"neutral": 1.0}
                    confidence = float(txt_scores.get(txt_label, 0.0))
                    if confidence < CONF_THRESHOLD:
                        txt_label = "neutral"
                    # Generate a reply locally (use your chatbot templates)
                    reply = generate_reply(txt_label, user_text)
                except Exception as e:
                    txt_label = "neutral"
                    confidence = 0.0
                    reply = generate_reply("neutral", user_text)

                # append to session history (store with human-readable ts)
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                st.session_state.chat_history.append({"who": "You", "text": user_text, "emotion": txt_label, "confidence": confidence, "ts": ts})
                st.session_state.chat_history.append({"who": "Bot", "text": reply, "emotion": txt_label, "confidence": confidence, "ts": ts})
                # clear input
                st.session_state["input_text"] = ""

                # rerun to refresh UI state
                st.experimental_rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# --- Bottom controls: download conversation ---
if st.sidebar.button("Download conversation"):
    # export recent history to CSV
    hist = st.session_state.get("chat_history", [])
    if not hist:
        st.sidebar.info("No conversation to download.")
    else:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["who", "text", "emotion", "confidence", "timestamp"])
        for item in hist:
            writer.writerow([item.get("who"), item.get("text"), item.get("emotion"), item.get("confidence"), item.get("ts")])
        st.sidebar.download_button("Download CSV", data=output.getvalue(), file_name="conversation.csv", mime="text/csv")

# small footer
st.markdown("<div style='margin-top:14px; font-size:12px; color:#6b7280'>Built with ❤️ — local emotion detection • keep models in <code>models/</code></div>", unsafe_allow_html=True)
