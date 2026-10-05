import re
import nltk
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

# Download required NLTK resources automatically if missing
resources = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("chunkers/maxent_ne_chunker_tab", "maxent_ne_chunker_tab"),
    ("corpora/words", "words"),
]

for path, package in resources:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(package, quiet=True)


def clean_text(text):
    """Basic text cleaning."""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_named_entities(sentence):
    """Extract named entities such as people, places and organizations."""
    try:
        tokens = word_tokenize(sentence)
        tagged = pos_tag(tokens)
        tree = ne_chunk(tagged)

        entities = []
        for subtree in tree:
            if isinstance(subtree, Tree):
                entity = " ".join(word for word, tag in subtree.leaves())
                if entity:
                    entities.append(entity)
        return entities
    except Exception:
        return []


def generate_questions(text, max_questions=8):
    """
    Generate simple questions using NLP preprocessing,
    POS tagging, named entities and rule-based templates.
    """
    text = clean_text(text)

    if not text:
        return []

    sentences = sent_tokenize(text)
    questions = []

    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tagged = pos_tag(tokens)

        # Remove punctuation for easier matching
        words = [w for w in tokens if re.search(r"[A-Za-z0-9]", w)]

        if len(words) < 4:
            continue

        lower = [w.lower() for w in words]

        # 1. Definition pattern:
        # "X is a Y" / "X is an Y" / "X is the Y"
        match = re.match(
            r"(.+?)\s+(?:is|are)\s+(?:a|an|the)\s+(.+?)(?:[.!?]|$)",
            sentence.strip(),
            re.IGNORECASE
        )
        if match:
            subject = match.group(1).strip()
            questions.append(f"What is {subject}?")

        # 2. "allows/enables/helps" pattern
        for verb in ["allows", "enables", "helps", "provides", "supports", "uses"]:
            if verb in lower:
                idx = lower.index(verb)
                subject = " ".join(words[:idx])
                if subject:
                    questions.append(f"What does {subject} {verb}?")

        # 3. Named entity question
        entities = get_named_entities(sentence)
        if entities:
            entity = entities[0]
            if entity.lower() not in {"machine learning", "artificial intelligence"}:
                questions.append(f"What is {entity}?")

        # 4. Predefined question from important noun
        nouns = [word for word, tag in tagged
                 if tag.startswith(("NN", "NNP")) and len(word) > 3]

        if nouns:
            noun = nouns[0]
            questions.append(f"What is the role of {noun}?")

        # 5. "used for" pattern
        used_match = re.search(
            r"(.+?)\s+is\s+used\s+for\s+(.+?)(?:[.!?]|$)",
            sentence,
            re.IGNORECASE
        )
        if used_match:
            subject = used_match.group(1).strip()
            purpose = used_match.group(2).strip()
            questions.append(f"What is {subject} used for?")
            questions.append(f"Why is {subject} used for {purpose}?")

        if len(questions) >= max_questions * 2:
            break

    # Remove duplicates while preserving order
    unique_questions = []
    seen = set()

    for q in questions:
        q = re.sub(r"\s+", " ", q).strip()
        key = q.lower()
        if key not in seen:
            seen.add(key)
            unique_questions.append(q)

    return unique_questions[:max_questions]
