"""Area B (classical NLP): regex normalization, model-code extraction, minimum edit distance
spelling correction, and ELIZA-style rule-based small-talk replies."""
import re
from collections import Counter

URL = re.compile(r"https?://\S+|www\.\S+")
# product model codes such as DW-6534, wm1200, RF-20481 (1-3 letters + optional hyphen + 3-5 digits)
CODE = re.compile(r"\b([A-Za-z]{1,3})-?(\d{3,5})\b")

APPLIANCES = ["dishwasher", "refrigerator", "fridge", "freezer", "washing", "washer", "dryer",
              "oven", "microwave", "cooktop", "stove", "kettle", "vacuum", "purifier", "humidifier",
              "conditioner", "thermostat", "blender", "toaster", "coffee", "machine", "drain",
              "drains", "leaking", "error", "warranty", "filter", "repair", "technician"]


def extract_codes(text: str):
    return [f"{a.upper()}-{b}" for a, b in CODE.findall(text)]


def normalize(text: str) -> str:
    """URL removal, model-code masking, lowercasing, repeated-letter squeeze, punctuation cleanup."""
    t = URL.sub(" ", text)
    t = CODE.sub(" modelcode ", t)
    t = t.lower()
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)          # sooooo -> soo
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def edit_distance(a: str, b: str, sub_cost: int = 1) -> int:
    """Minimum edit distance by dynamic programming (insert=delete=1, substitute=sub_cost)."""
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if a[i - 1] == b[j - 1] else sub_cost))
        prev = cur
    return prev[m]


def build_lexicon(train_texts, min_freq: int = 2):
    counts = Counter(w for t in train_texts for w in normalize(t).split())
    lex = {w: c for w, c in counts.items() if c >= min_freq and w.isalpha()}
    for w in APPLIANCES:
        lex[w] = lex.get(w, 0) + min_freq
    return lex


def correct_text(text: str, lexicon: dict):
    """Replace out-of-lexicon words (len>=4) by the closest lexicon word within edit distance
    1 (short words) or 2 (long words); ties are broken by corpus frequency.
    Returns (corrected_text, [(wrong, right), ...])."""
    out, fixes = [], []
    for w in text.split():
        if len(w) < 4 or w in lexicon or not w.isalpha():
            out.append(w); continue
        limit = 1 if len(w) <= 6 else 2
        best, best_key = None, None
        for cand, freq in lexicon.items():
            if abs(len(cand) - len(w)) > limit:
                continue
            d = edit_distance(w, cand)
            if d <= limit and (best_key is None or (d, -freq) < best_key):
                best, best_key = cand, (d, -freq)
        if best:
            fixes.append((w, best)); out.append(best)
        else:
            out.append(w)
    return " ".join(out), fixes


# ---- ELIZA-style rules (regex pattern -> reply template with capture groups) -------------------
ELIZA_RULES = [
    (r"^\s*(hi|hello|hey|good (morning|afternoon|evening))\b[\s\w!.,]{0,25}$",
     "Hello! I'm the support assistant. Tell me what's wrong with your appliance."),
    (r"\b(thank you|thanks)\b", "You're welcome! Is there anything else I can help with?"),
    (r"^\s*(bye|goodbye|see you)\b", "Goodbye! Have a nice day."),
    (r"\bhow are you\b", "I'm just a program, but I'm ready to help. What can I do for you?"),
    (r"\bmy name is (\w+)", "Nice to meet you, \\1! How can I help you today?"),
    (r"\bi(?: am|'m) (sad|angry|upset|frustrated|annoyed)",
     "I am sorry to hear you are \\1. Can you describe what happened?"),
    (r"\b(always|never)\b.{0,40}\b(break|breaks|broken|fail|fails)\b",
     "Can you think of a specific example?"),
]


def eliza_reply(text: str):
    for pattern, template in ELIZA_RULES:
        m = re.search(pattern, text, flags=re.I)
        if m:
            return m.expand(template)
    return None
