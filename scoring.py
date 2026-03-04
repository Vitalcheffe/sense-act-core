import logging

log = logging.getLogger(__name__)

_pipeline = None
_ready = False


def _load():
    global _pipeline, _ready
    if _pipeline is not None:
        return
    try:
        from transformers import pipeline as hf_pipeline
        _pipeline = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            max_length=512,
            truncation=True,
        )
        _ready = True
        log.info("FinBERT loaded")
    except Exception as e:
        log.warning("FinBERT unavailable, using keywords: %s", e)
        _ready = False


_BULL = [
    "cut", "shortage", "rally", "surge", "bullish", "deficit", "strong",
    "increase", "rise", "gain", "production cut", "underproduction",
    "outperform", "upgrade", "beat", "record", "sanctions", "disruption",
    "tighten", "drawdown", "stockpile drop", "below forecast",
]
_BEAR = [
    "glut", "crash", "bearish", "oversupply", "drop", "weak",
    "flood", "sell", "decline", "fall", "surplus", "output rise",
    "downgrade", "miss", "inventory build", "demand slump", "recession",
    "above forecast", "stockpile rise", "production increase",
]


def _keyword(text):
    words = [w.strip(".,!?;:\"'()") for w in text.lower().split()]
    p = sum(1 for w in words if w in _BULL)
    n = sum(1 for w in words if w in _BEAR)
    return (p - n) / (p + n) if (p + n) > 0 else 0.0


def score(text: str) -> float:
    _load()
    if not text or not text.strip():
        return 0.0
    if _ready and _pipeline is not None:
        try:
            r = _pipeline(text[:512])[0]
            s = float(r["score"])
            label = r["label"].lower()
            if label == "positive":
                return s
            if label == "negative":
                return -s
            return 0.0
        except Exception as e:
            log.warning("FinBERT error: %s", e)
    return _keyword(text)


def score_batch(texts: list) -> list:
    _load()
    if not texts:
        return []
    if _ready and _pipeline is not None:
        try:
            results = _pipeline([t[:512] for t in texts], batch_size=16)
            out = []
            for r in results:
                s = float(r["score"])
                label = r["label"].lower()
                out.append(s if label == "positive" else -s if label == "negative" else 0.0)
            return out
        except Exception as e:
            log.warning("FinBERT batch error: %s", e)
    return [_keyword(t) for t in texts]


def finbert_active() -> bool:
    _load()
    return _ready
