"""
SmartResume AI - Named Entity Recognition (NER) Module
Extracts resume-specific entities: PERSON, EMAIL, PHONE, ORGANIZATION,
LOCATION, DATE, DEGREE, UNIVERSITY, SKILL, JOB_TITLE, CERTIFICATION, PROJECT, COMPANY.
Integrates trained transition NER, EntityRuler patterns, and regex validators.
"""

import re
import json
import html
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

import spacy
from config.settings import MODELS_DIR, SKILL_KB_PATH

logger = logging.getLogger(__name__)

TRAINED_MODEL_PATH = MODELS_DIR / "resume_ner_model"

# Color palette for entity badges in light theme
ENTITY_COLORS = {
    "PERSON": "#E0E7FF",        # Indigo tint
    "EMAIL": "#FEF3C7",         # Amber tint
    "PHONE": "#FEE2E2",         # Red tint
    "ORGANIZATION": "#DBEAFE",  # Blue tint
    "LOCATION": "#E0F2FE",      # Sky tint
    "DATE": "#F3E8FF",          # Purple tint
    "DEGREE": "#ECFCCB",        # Lime tint
    "UNIVERSITY": "#D1FAE5",    # Emerald tint
    "SKILL": "#CCFBF1",         # Teal tint
    "JOB_TITLE": "#EDE9FE",     # Violet tint
    "CERTIFICATION": "#FCE7F3", # Pink tint
    "PROJECT": "#FEF9C3",       # Yellow tint
    "COMPANY": "#E2E8F0"        # Slate tint
}

ENTITY_TEXT_COLORS = {
    "PERSON": "#3730A3",
    "EMAIL": "#92400E",
    "PHONE": "#991B1B",
    "ORGANIZATION": "#1E40AF",
    "LOCATION": "#075985",
    "DATE": "#6B21A8",
    "DEGREE": "#3F6212",
    "UNIVERSITY": "#065F46",
    "SKILL": "#115E59",
    "JOB_TITLE": "#5B21B6",
    "CERTIFICATION": "#9D174D",
    "PROJECT": "#854D0E",
    "COMPANY": "#334155"
}


class ResumeNER:
    """Hybrid Resume Named Entity Recognition Engine."""

    def __init__(self):
        self.nlp = self._load_or_build_pipeline()

    def _load_or_build_pipeline(self):
        """Loads trained resume model or constructs rule-based pipeline."""
        if TRAINED_MODEL_PATH.exists():
            try:
                nlp = spacy.load(TRAINED_MODEL_PATH)
                logger.info(f"Loaded trained NER model from {TRAINED_MODEL_PATH}")
                return nlp
            except Exception as e:
                logger.warning(f"Could not load trained model: {e}. Building fallback pipeline.")

        # Fallback English pipeline
        nlp = spacy.blank("en")
        ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})

        patterns = []
        if SKILL_KB_PATH.exists():
            try:
                with open(SKILL_KB_PATH, "r", encoding="utf-8") as f:
                    kb = json.load(f)
                    for item in kb.get("skills", []):
                        patterns.append({"label": "SKILL", "pattern": item["canonical_name"]})
                        for alias in item.get("aliases", []):
                            patterns.append({"label": "SKILL", "pattern": alias})
            except Exception:
                pass

        common_titles = [
            "Software Engineer", "Senior Software Engineer", "Data Scientist",
            "Machine Learning Engineer", "DevOps Engineer", "Product Manager",
            "Full Stack Developer", "Cloud Architect", "Project Manager"
        ]
        for t in common_titles:
            patterns.append({"label": "JOB_TITLE", "pattern": t})

        ruler.add_patterns(patterns)
        return nlp

    @staticmethod
    def _sanitize_entity_span(text: str, start: int, end: int, label: str) -> Optional[Tuple[str, int, int, str]]:
        """
        Sanitizes raw entity spans by removing leading/trailing punctuation and whitespace,
        re-aligning start/end character offsets, and filtering out punctuation-only or invalid tokens.
        """
        strip_chars = ' \t\r\n,;:•*–—/\\|&()[]{}<>"\'`'
        s = start
        e = end
        while s < e and text[s] in strip_chars:
            s += 1
        while e > s and text[e - 1] in strip_chars:
            e -= 1

        clean_text = text[s:e].strip()
        if not clean_text or not re.search(r'[A-Za-z0-9]', clean_text):
            return None

        # Filter out noisy or incomplete entity fragments
        if label == "SKILL":
            if clean_text.lower() in ("/", "-", "&", "and", "the", "or", "for", "with", "in", "expe", "expe-"):
                return None
            if len(clean_text) == 1 and clean_text.upper() not in ("C", "R"):
                return None
            if clean_text.endswith("-"):
                return None
        elif label == "LOCATION":
            if len(clean_text) < 2:
                return None
        elif label == "EXPERIENCE":
            # If the model labeled a pure date range as EXPERIENCE, relabel to DATE
            if re.search(r'\b(?:19\d{2}|20\d{2})\b', clean_text):
                label = "DATE"

        return clean_text, s, e, label

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Runs hybrid NER: SpaCy pipeline + Regex patterns for contacts/links.
        Returns deduplicated, sorted entity list with spans and confidence scores.
        """
        if not text:
            return []

        entities = []
        occupied_spans = []

        # 1. Regex for Email
        email_regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b')
        for match in email_regex.finditer(text):
            entities.append({
                "text": match.group(),
                "label": "EMAIL",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.99
            })
            occupied_spans.append((match.start(), match.end()))

        # 2. Regex for Phone
        phone_regex = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
        for match in phone_regex.finditer(text):
            entities.append({
                "text": match.group(),
                "label": "PHONE",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.95
            })
            occupied_spans.append((match.start(), match.end()))

        # 3. Model Entities with sanitization
        doc = self.nlp(text)
        for ent in doc.ents:
            res = self._sanitize_entity_span(text, ent.start_char, ent.end_char, ent.label_)
            if not res:
                continue
            clean_text, s, e, sanitized_label = res

            # Avoid collision with verified regex spans
            collision = any(not (e <= occ_s or s >= occ_e) for occ_s, occ_e in occupied_spans)
            if not collision:
                entities.append({
                    "text": clean_text,
                    "label": sanitized_label,
                    "start": s,
                    "end": e,
                    "confidence": 0.90
                })
                occupied_spans.append((s, e))

        # Sort entities by start position
        entities = sorted(entities, key=lambda x: (x["start"], -x["end"]))

        # Deduplicate identical spans
        deduped = []
        seen = set()
        for ent in entities:
            key = (ent["start"], ent["end"])
            if key not in seen:
                seen.add(key)
                deduped.append(ent)

        return deduped

    @staticmethod
    def render_highlighted_text(text: str, entities: List[Dict[str, Any]]) -> str:
        """
        Generates clean HTML with colored highlight pills for Streamlit preview.
        """
        if not text:
            return ""

        sorted_ents = sorted(entities, key=lambda x: x["start"])
        html_parts = []
        last_idx = 0

        for ent in sorted_ents:
            s, e = ent["start"], ent["end"]
            if s < last_idx or s > len(text) or e > len(text):
                continue

            # Append plain text preceding entity (HTML escaped)
            html_parts.append(html.escape(text[last_idx:s]).replace("\n", "<br>"))

            bg = ENTITY_COLORS.get(ent["label"], "#F1F5F9")
            fg = ENTITY_TEXT_COLORS.get(ent["label"], "#334155")
            ent_label = html.escape(ent["label"])
            ent_text = html.escape(text[s:e])

            badge = (
                f'<mark style="background-color: {bg}; color: {fg}; padding: 2px 6px; '
                f'border-radius: 4px; font-weight: 500; margin: 0 2px; border: 1px solid rgba(0,0,0,0.05);">'
                f'{ent_text} <span style="font-size: 0.72em; opacity: 0.85; text-transform: uppercase;">'
                f'[{ent_label}]</span></mark>'
            )
            html_parts.append(badge)
            last_idx = e

        html_parts.append(html.escape(text[last_idx:]).replace("\n", "<br>"))
        return "".join(html_parts)

