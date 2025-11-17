# src/text_emotion.py

from transformers import pipeline

# Load sentiment/emotion classifier
classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

def analyze_text(text):
    """
    Returns: 
    - top emotion label (str)
    - a dict: {emotion: score}
    """

    results = classifier(text)

    # results is a LIST containing one element: a LIST of dicts
    # We take the first element properly
    scores_list = results[0]  # list of dicts: [{'label': 'joy', 'score': 0.9}, ...]

    # Convert to a dictionary
    scores = {item["label"].lower(): float(item["score"]) for item in scores_list}

    # Pick top emotion
    top_label = max(scores, key=scores.get)

    return top_label, scores
