"""
gap_engine.py
Production Module: Research Gap Identification & Classification (Member 4)
Integrates with Member 2 Backend API (/identify-gap)
"""

import re
import unicodedata
import fitz  # PyMuPDF
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("[Member 4 Engine] Initializing SciBERT/SBERT Inference Engine...")
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

TAXONOMY = {
    "Methodological Gap": "algorithmic approach architecture limits baseline design flaws loss function gradient vanishing optimization trade-off",
    "Dataset Gap": "small dataset size lack domain diversity benchmark evaluation bias low quality training data annotation scarcity",
    "Evaluation Gap": "inadequate metrics missing human evaluation lack statistical testing missing baseline comparison",
    "Application Gap": "real-world deployment latency memory footprint compute resource constraint edge device inference cost",
    "Future Scope Gap": "unexplored downstream tasks future work proposed extensions cross-domain applications open directions"
}

category_names = list(TAXONOMY.keys())
category_embeddings = embedder.encode(list(TAXONOMY.values()))

EXTRACTION_ANCHORS = {
    "contributions": "in this paper we propose introduce present develop our main contribution architecture",
    "methodology": "we train implement design loss function neural network fine-tuned hyperparameters",
    "limitations": "limitation drawback constraint fail underperform bottleneck inferior trade-off computational cost",
    "future_work": "in the future future work further research remains open direction we plan to extend"
}
anchor_embeddings = {k: embedder.encode(v) for k, v in EXTRACTION_ANCHORS.items()}


def clean_pdf(doc) -> str:
    pages = []
    total = len(doc)
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if i > int(total * 0.75):
            ref = re.search(r'\n(References|BIBLIOGRAPHY|REFERENCES)\n', text)
            if ref:
                pages.append(text[:ref.start()])
                break
        pages.append(text)
    full = unicodedata.normalize("NFKD", "\n".join(pages))
    full = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', full)
    full = re.sub(r'[\u2022\u2013\u2014\u2012\u2019]', "'", full)
    return re.sub(r'\s+', ' ', full).strip()


def parse_discourse(clean_text: str) -> dict:
    doc = nlp(clean_text)
    candidates = []
    for sent in doc.sents:
        s = sent.text.strip()
        if 8 <= len(s.split()) <= 60 and not re.search(r'\[\d+\]|\(\d{4}\)|\bsection\b|\btable\b|\bfigure\b', s.lower()):
            candidates.append(s)

    if not candidates:
        return {"contributions": [], "methodology": [], "limitations": [], "future_work": []}

    sent_embs = embedder.encode(candidates)
    sections = {"contributions": [], "methodology": [], "limitations": [], "future_work": []}

    for key, anchor_emb in anchor_embeddings.items():
        sims = cosine_similarity(sent_embs, anchor_emb.reshape(1, -1)).flatten()
        for idx in sims.argsort()[::-1]:
            if sims[idx] >= 0.28:
                sections[key].append(candidates[idx])
            if len(sections[key]) >= 4:
                break

    for s in candidates:
        if any(k in s.lower() for k in ["we train", "we implement", "loss function", "fine-tuned", "hyperparameters"]):
            sections["methodology"].append(s)
        if len(sections["methodology"]) >= 4:
            break

    return sections


def build_recommendation(category: str) -> str:
    recs = {
        "Methodological Gap": "Explore alternative attention topologies or hybrid architectures to overcome algorithmic trade-offs.",
        "Dataset Gap": "Benchmark generalization across low-resource, out-of-domain, or specialized scientific corpora.",
        "Evaluation Gap": "Conduct comprehensive validation using diverse human evaluators, significance testing, and broader domain benchmarks.",
        "Application Gap": "Investigate model quantization, structured pruning, and LoRA to reduce inference latency and memory footprint.",
        "Future Scope Gap": "Extend the proposed framework to multi-task transfer, edge streaming, or cross-modal environments."
    }
    return recs.get(category, "Further empirical investigation is recommended.")


def classify_gaps(gap_sentences: list) -> list:
    if not gap_sentences:
        return []
    unique_sents = list(dict.fromkeys(gap_sentences))
    sent_embs = embedder.encode(unique_sents)
    sims = cosine_similarity(sent_embs, category_embeddings)

    gaps = []
    for i, s in enumerate(unique_sents):
        best_idx = sims[i].argmax()
        cat = category_names[best_idx]
        conf = float(sims[i][best_idx])
        gaps.append({
            "gap_category": cat,
            "confidence_score": round(conf, 3),
            "statement": s,
            "recommendation": build_recommendation(cat)
        })
    return sorted(gaps, key=lambda x: x["confidence_score"], reverse=True)


def identify_research_gaps(pdf_path_or_bytes) -> dict:
    """Primary function called by Member 2 Backend API."""
    if isinstance(pdf_path_or_bytes, bytes):
        doc = fitz.open(stream=pdf_path_or_bytes, filetype="pdf")
    else:
        doc = fitz.open(pdf_path_or_bytes)

    clean_text = clean_pdf(doc)
    discourse = parse_discourse(clean_text)
    gap_pool = discourse["limitations"] + discourse["future_work"]
    ranked_gaps = classify_gaps(gap_pool)

    return {
        "contributions": discourse["contributions"][:3],
        "methodology": discourse["methodology"][:3],
        "limitations": discourse["limitations"][:3],
        "future_work": discourse["future_work"][:3],
        "identified_gaps": ranked_gaps[:4]
    }