# sentiment.py
# trying to build something that reads oil news and figures out if it's bullish or bearish
# started this after reading about quantitative finance on reddit lol
# very basic for now, will improve

import time

# keywords that usually mean price goes up
BULLISH = ["cut", "shortage", "rally", "surge", "buy", "up", "increase", "deficit", "bullish"]
# keywords that usually mean price goes down  
BEARISH = ["glut", "drop", "sell", "down", "decrease", "crash", "oversupply", "bearish"]


def get_sentiment(text):
    words = text.lower().split()
    
    pos = 0
    neg = 0
    
    for word in words:
        # strip punctuation manually (TODO: use regex later)
        word = word.strip(".,!?;:")
        if word in BULLISH:
            pos += 1
        if word in BEARISH:
            neg += 1
    
    # this breaks if text is empty, fix later
    score = (pos - neg) / len(words)
    return score


def process_signal(tweet_text, source, followers):
    score = get_sentiment(tweet_text)
    
    print(f"[{source}] followers={followers}")
    print(f"  text: {tweet_text[:80]}")
    print(f"  sentiment score: {score:.4f}")
    print(f"  direction: {'BULLISH' if score > 0 else 'BEARISH' if score < 0 else 'NEUTRAL'}")
    print()
    
    return score


# test with some fake tweets
if __name__ == "__main__":
    signals = [
        ("Reuters",         2000000, "OPEC cuts oil production, supply deficit expected, bullish outlook"),
        ("oil_trader_anon", 3500,    "massive crash incoming, oversupply everywhere, very bearish"),
        ("noise_account",   50,      "just had my morning coffee"),
        ("aramco_engineer", 200,     "pipeline maintenance will cause supply shortage next week"),
        ("generic_bot",     800000,  "oil price up today, bullish"),
    ]
    
    print("=== SENSE-ACT v0.1 ===")
    print()
    
    results = []
    for source, followers, text in signals:
        score = process_signal(text, source, followers)
        results.append((source, followers, score))
    
    print("--- Summary ---")
    for source, followers, score in results:
        print(f"  {source}: {score:+.4f}")
