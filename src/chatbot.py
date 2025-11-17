# src/chatbot.py
import random
from typing import Optional

# Tone templates by emotion. Short, conversational, and varied.
TEMPLATES = {
    "happy": [
        "That's awesome — congrats! 🎉 What about it made you happiest?",
        "Great news! I'm so happy for you. Want to tell me more?",
        "Love hearing that — sounds like a win! How do you want to celebrate?"
    ],
    "joy": [
        "That's awesome — congrats! 🎉 What about it made you happiest?",
        "You sound excited — tell me more!",
    ],
    "sad": [
        "I'm sorry you're feeling down. I'm here to listen — do you want to talk about it?",
        "That sounds tough. Would you like to share more about what's going on?",
        "I hear you — it's okay to feel this way. What happened?"
    ],
    "sadness": [
        "I'm sorry you're feeling down. I'm here to listen — do you want to talk about it?",
        "That sounds tough. Would you like to share more about what's going on?"
    ],
    "angry": [
        "I get why you're upset — that sounds frustrating. Want to tell me what happened?",
        "That must be annoying. Do you want help thinking through it or a distraction?"
    ],
    "fear": [
        "That sounds worrying. Do you want to talk it through or get some calming tips?",
        "I’m here — tell me what’s making you anxious and we’ll take it step by step."
    ],
    "neutral": [
        "Okay — tell me more when you're ready.",
        "Got it. Anything else on your mind?"
    ],
    "surprise": [
        "Oh wow — that's surprising! Tell me more.",
        "Whoa — that's unexpected. What happened next?"
    ],
    "disgust": [
        "That sounds unpleasant — I'm sorry you experienced that. Want to talk about it?",
        "Ugh — that does sound gross. Do you want suggestions to handle it?"
    ],
    "love": [
        "That’s lovely to hear! Tell me more about it.",
        "Warm and fuzzy — that makes me smile for you. Share more?"
    ]
}

# Gentle mapping from predicted labels to our template keys
LABEL_MAP = {
    "joy": "joy",
    "happy": "joy",
    "sad": "sad",
    "sadness": "sad",
    "angry": "angry",
    "anger": "angry",
    "fear": "fear",
    "neutral": "neutral",
    "surprise": "surprise",
    "disgust": "disgust",
    "love": "love"
}

def _choose_template(label: str) -> str:
    """Pick a random template for the given label, fallback to neutral."""
    label = (label or "neutral").lower()
    key = LABEL_MAP.get(label, "neutral")
    choices = TEMPLATES.get(key, TEMPLATES["neutral"])
    return random.choice(choices)

def generate_reply(emotion_label: Optional[str], user_text: Optional[str] = None) -> str:
    """
    Return a conversational reply based on emotion_label.
    user_text is optionally echoed briefly to show understanding.
    """
    template = _choose_template(emotion_label)
    # keep the echoed text short
    echo = (user_text or "").strip()
    if echo:
        # Only echo a short piece (avoid long repetition)
        echo_snip = echo if len(echo) <= 120 else echo[:117] + "..."
        return f"{template} (You said: '{echo_snip}')"
    return template
