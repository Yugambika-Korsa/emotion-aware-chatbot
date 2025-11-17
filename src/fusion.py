# src/fusion.py
def fuse(audio_label, audio_scores, text_label, text_scores, weights=(0.6, 0.4)):
    """
    Simple fusion of audio and text emotion scores.

    Parameters
    ----------
    audio_label : str
        Best label predicted from audio (e.g., "happy").
    audio_scores : dict
        Mapping label -> score from audio model (e.g., {"happy":0.7, "sad":0.2}).
    text_label : str
        Best label predicted from text (e.g., "joy").
    text_scores : dict
        Mapping label -> score from text model (e.g., {"joy":0.8, "anger":0.1}).
    weights : tuple(float, float)
        (audio_weight, text_weight) how much to trust each modality.

    Returns
    -------
    final_label : str
        The fused label with highest combined score.
    combined_scores : dict
        Mapping label -> combined score.
    """
    audio_w, text_w = weights
    combined = {}

    # Ensure label keys are lowercase strings
    audio_scores = {str(k).lower(): float(v) for k, v in (audio_scores or {}).items()}
    text_scores = {str(k).lower(): float(v) for k, v in (text_scores or {}).items()}
    audio_label = str(audio_label).lower() if audio_label else None
    text_label = str(text_label).lower() if text_label else None

    # include the single-best labels even if score dicts are empty
    labels = set(list(audio_scores.keys()) + list(text_scores.keys()))
    if audio_label:
        labels.add(audio_label)
    if text_label:
        labels.add(text_label)

    for l in labels:
        a = audio_scores.get(l, 0.0)
        t = text_scores.get(l, 0.0)
        combined[l] = audio_w * a + text_w * t

    # fallback to neutral if nothing present
    if not combined:
        return "neutral", {"neutral": 1.0}

    final = max(combined, key=combined.get)
    return final, combined
