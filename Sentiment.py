import hashlib
import math
import time
from dataclasses import dataclass


BULLISH = [
    "cut", "shortage", "rally", "surge", "buy", "increase",
    "deficit", "bullish", "strong", "underproduction"
]
BEARISH = [
    "glut", "drop", "sell", "decrease", "crash",
    "oversupply", "bearish", "weak", "flood"
]


@dataclass
class Signal:
    uid: str
    source: str
    followers: int
    text: str
    raw_score: float
    weighted_score: float
    direction: str
    timestamp: float
    is_duplicate: bool = False


def keyword_score(text):
    if not text or not text.strip():
        return 0.0

    words = [w.strip(".,!?;:") for w in text.lower().split()]
    pos = sum(1 for w in words if w in BULLISH)
    neg = sum(1 for w in words if w in BEARISH)
    total = pos + neg

    if total == 0:
        return 0.0

    return (pos - neg) / total


def follower_weight(n):
    # not sure log is the right scale here but linear felt wrong
    # reuters at 2M was completely drowning out everyone else
    return min(math.log10(max(n, 1) + 1) / 7.0, 1.0)


class SignalProcessor:
    def __init__(self):
        self._seen = set()

    def process(self, text, source, followers):
        uid = hashlib.md5(text.lower().strip().encode()).hexdigest()[:8]
        h = hashlib.md5(text.lower().strip().encode()).hexdigest()

        is_dup = h in self._seen
        if not is_dup:
            self._seen.add(h)

        raw = keyword_score(text)
        w = follower_weight(followers)
        weighted = raw * w

        if weighted > 0.08:
            direction = "BULLISH"
        elif weighted < -0.08:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        return Signal(
            uid=uid,
            source=source,
            followers=followers,
            text=text,
            raw_score=raw,
            weighted_score=weighted,
            direction=direction,
            timestamp=time.time(),
            is_duplicate=is_dup,
        )


if __name__ == "__main__":
    proc = SignalProcessor()

    cases = [
        ("Reuters",         2_000_000, "OPEC cuts oil production, supply deficit bullish"),
        ("oil_trader_anon", 3_500,     "massive crash incoming, oversupply bearish"),
        ("noise_account",   50,        "just had my morning coffee"),
        ("aramco_engineer", 200,       "pipeline maintenance will cause supply shortage"),
        ("generic_bot_1",   800_000,   "oil price up today bullish"),
        ("generic_bot_2",   750_000,   "oil price up today bullish"),
        ("empty",           1_000,     ""),
    ]

    print("=== sense-act ===\n")

    for source, followers, text in cases:
        s = proc.process(text, source, followers)
        flag = " [dup]" if s.is_duplicate else ""
        print(f"{source}{flag}")
        print(f"  raw={s.raw_score:+.3f}  w={follower_weight(followers):.2f}  "
              f"weighted={s.weighted_score:+.3f}  {s.direction}")
