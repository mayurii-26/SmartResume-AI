"""
SmartResume AI - Grounded GroqCloud Generative Assistant Module
Provides evidence-grounded resume Q&A, executive summaries, optimization suggestions,
and structured insights powered by the official GroqCloud API (https://api.groq.com/openai/v1).
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional
import requests

from config.settings import GROQ_API_KEY, GROQ_MODEL, GROQ_API_BASE

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are SmartResume AI, an evidence-grounded resume assistant.
Use only facts explicitly present in the uploaded resume text. Never invent a
qualification, employer, date, skill, degree, project, certification, or contact
detail. When information is unavailable, say exactly: "Not mentioned in the
uploaded resume". Keep each response limited to the supplied resume."""

DEFAULT_GROQ_MODEL = "qwen/qwen3.8-27b"

SUPPORTED_GROQ_MODELS = (
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "allam-2-7b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
)


class GenAIAssistant:
    """Production-grade client for the official GroqCloud API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ):
        raw_key = api_key if api_key is not None else (
            os.getenv("GROQ_API_KEY", "") or GROQ_API_KEY
        )
        self.api_key = raw_key.strip() if raw_key else ""
        raw_model = model_name if model_name is not None else (
            os.getenv("GROQ_MODEL", "") or GROQ_MODEL or DEFAULT_GROQ_MODEL
        )
        self.model_name = raw_model.strip() if raw_model else DEFAULT_GROQ_MODEL
        raw_base = base_url if base_url is not None else (
            os.getenv("GROQ_API_BASE", "") or GROQ_API_BASE
        )
        self.base_url = raw_base.rstrip("/") if raw_base else "https://api.groq.com/openai/v1"
        self._session = session or requests.Session()
        self._cached_available_models: Optional[List[str]] = None

    def is_configured(self) -> bool:
        """Checks if a valid, non-placeholder Groq API key is available."""
        return bool(self.api_key and not self.api_key.startswith("your_"))

    def configuration_message(self) -> str:
        """Returns a user-friendly instruction for setting up the API key."""
        if not self.api_key or self.api_key.startswith("your_"):
            return "Groq is not configured. Add GROQ_API_KEY to your local .env file or Settings and restart the app."
        return "Groq is unavailable. Check the local API configuration and try again."

    def get_available_models(self, force_refresh: bool = False) -> List[str]:
        """Dynamically retrieves active chat/completion models from GroqCloud API.
        Filters out non-text/audio/guard models. Falls back to verified list if offline.
        """
        if self._cached_available_models and not force_refresh:
            return list(self._cached_available_models)

        if not self.is_configured():
            return list(SUPPORTED_GROQ_MODELS)

        try:
            resp = self._session.get(
                f"{self.base_url}/models",
                headers=self._get_headers(),
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                excluded_patterns = ("whisper", "prompt-guard", "guard", "orpheus", "distil-whisper", "tts", "playai")
                chat_models = []
                for item in data:
                    model_id = item.get("id", "")
                    active = item.get("active", True)
                    if not active or not model_id:
                        continue
                    if any(pat in model_id.lower() for pat in excluded_patterns):
                        continue
                    chat_models.append(model_id)

                if chat_models:
                    self._cached_available_models = sorted(chat_models)
                    return list(self._cached_available_models)
        except Exception as e:
            logger.warning(f"Could not dynamically query Groq models: {e}")

        return list(SUPPORTED_GROQ_MODELS)

    @staticmethod
    def _error_message(status_code: Optional[int], message: str, model_name: str = "") -> str:
        """Translates API and HTTP errors into clear, actionable, sanitized messages.
        Ensures raw API keys are never exposed in logs or user-facing messages.
        """
        lower = message.lower()
        if status_code == 401 or any(token in lower for token in ("unauthorized", "invalid_api_key", "unauthenticated", "invalid api key")):
            return "Groq could not authenticate the API key. Please check GROQ_API_KEY in your .env or Settings."
        if status_code in (402, 403) or any(token in lower for token in ("forbidden", "permission_denied", "access_denied")):
            return "Access forbidden or insufficient permissions. Check your GroqCloud account settings and API key permissions."
        if status_code == 429 or any(token in lower for token in ("rate_limit_exceeded", "rate limit", "too many requests", "tokens per minute")):
            return "Groq rate limit reached (requests or tokens per minute). Please wait a moment and try again."
        if status_code == 404 or any(token in lower for token in ("model_not_found", "model unavailable", "does not exist")):
            if model_name:
                return f"The configured Groq model '{model_name}' is not accessible on your GroqCloud account. Please select an active model in Settings."
            return "The configured Groq model is unavailable or invalid. Select a supported model in Settings."
        if any(token in lower for token in ("timeout", "connection", "network", "offline", "dns", "max retries")):
            return "Groq API service is temporarily unreachable. Check your internet connection and try again."
        if "empty response" in lower:
            return "Groq returned an empty response. Please try again."
        return "Groq returned an unexpected error. Please check your request and try again."

    def _get_headers(self) -> Dict[str, str]:
        """Constructs safe authorization headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "SmartResumeAI/2.0",
        }

    def test_connection(self) -> Dict[str, Any]:
        """Tests live connectivity to the Groq API endpoint."""
        if not self.is_configured():
            return {"success": False, "error": self.configuration_message()}

        endpoint = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": "Reply with exactly: OK"}
            ],
            "max_tokens": 10,
            "temperature": 0.0,
        }

        try:
            response = self._session.post(
                endpoint,
                headers=self._get_headers(),
                json=payload,
                timeout=15,
            )
            if response.status_code == 200:
                data = response.json()
                text = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if not text:
                    raise RuntimeError("Groq returned an empty response")
                return {"success": True, "model": self.model_name}

            if response.status_code == 404:
                available = self.get_available_models(force_refresh=True)
                avail_str = ", ".join(available[:4]) if available else "qwen/qwen3.8-27b"
                return {
                    "success": False,
                    "error": (
                        f"The configured Groq model '{self.model_name}' is not accessible on your Groq account. "
                        f"Available models include: {avail_str}. Please select a supported model in Settings."
                    ),
                }

            return {
                "success": False,
                "error": self._error_message(response.status_code, response.text, self.model_name),
            }
        except requests.RequestException as e:
            logger.warning(f"Groq API connection test error: {e}")
            return {"success": False, "error": self._error_message(None, str(e), self.model_name)}
        except Exception as e:
            logger.warning(f"Unexpected Groq API error during connection test: {e}")
            return {"success": False, "error": self._error_message(None, str(e), self.model_name)}

    def _generate(
        self,
        resume_text: str,
        prompt: str,
        *,
        json_output: bool = False,
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_tokens: int = 1500,
    ) -> Dict[str, Any]:
        """Sends an evidence-grounded prompt to Groq via GroqCloud's chat completions endpoint."""
        if not self.is_configured():
            return {"success": False, "error": self.configuration_message(), "text": ""}

        endpoint = f"{self.base_url}/chat/completions"
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # Include prior conversation turns if provided
        if chat_history:
            for turn in chat_history[-4:]:
                if "question" in turn and "answer" in turn:
                    messages.append({"role": "user", "content": turn["question"]})
                    messages.append({"role": "assistant", "content": turn["answer"]})

        messages.append({
            "role": "user",
            "content": f"UPLOADED RESUME:\n---\n{resume_text}\n---\n\n{prompt}",
        })

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": max_tokens,
        }

        if json_output:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = self._session.post(
                endpoint,
                headers=self._get_headers(),
                json=payload,
                timeout=45,
            )
            if response.status_code == 200:
                data = response.json()
                text = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if not text:
                    raise RuntimeError("Groq returned an empty response")
                return {"success": True, "text": text, "model": self.model_name}

            if response.status_code == 404:
                available = self.get_available_models(force_refresh=True)
                avail_str = ", ".join(available[:4]) if available else "qwen/qwen3.8-27b"
                return {
                    "success": False,
                    "error": (
                        f"The configured Groq model '{self.model_name}' is not accessible on your Groq account. "
                        f"Available models include: {avail_str}. Please select an active model in Settings."
                    ),
                    "text": "",
                }

            return {
                "success": False,
                "error": self._error_message(response.status_code, response.text, self.model_name),
                "text": "",
            }
        except requests.RequestException as e:
            logger.warning(f"Groq API network request failed: {e}")
            return {"success": False, "error": self._error_message(None, str(e), self.model_name), "text": ""}
        except Exception as e:
            logger.warning(f"Groq API unexpected error: {e}")
            return {"success": False, "error": self._error_message(None, str(e), self.model_name), "text": ""}

    def answer_question(
        self,
        resume_text: str,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Answers a user inquiry strictly using facts extracted from the uploaded resume."""
        result = self._generate(
            resume_text,
            f"Answer this question about the uploaded resume: {question}",
            chat_history=chat_history,
        )
        return {**result, "answer": result.get("text", "")}

    def generate_summary(self, resume_text: str) -> Dict[str, Any]:
        """Generates an executive candidate summary strictly from uploaded resume facts."""
        prompt = (
            "Write a concise candidate summary with headings: "
            "Professional overview, Skills and education, Experience and projects. "
            "Cite only uploaded-resume facts."
        )
        result = self._generate(resume_text, prompt)
        return {**result, "summary": result.get("text", "")}

    def suggest_improvements(self, resume_text: str) -> Dict[str, Any]:
        """Provides actionable optimization guidance, metric improvements, and ATS keyword relevance."""
        prompt = (
            "Provide resume improvements under: Missing information, Impact and clarity, "
            "Formatting, and ATS keywords. Identify absent details as Not mentioned in the uploaded resume."
        )
        result = self._generate(resume_text, prompt)
        return {**result, "critique": result.get("text", "")}

    def generate_structured_insights(self, resume_text: str) -> Dict[str, Any]:
        """Produces strict, parseable JSON insights grounded exclusively in the resume."""
        prompt = (
            'Return JSON only with exactly: {"summary":"", "skills":[], "education":[], "experience":[], "projects":[], "improvements":[]}. '
            'Use strings in arrays. Use "Not mentioned in the uploaded resume" for unavailable scalar information '
            'and [] for unavailable list information.'
        )
        result = self._generate(resume_text, prompt, json_output=True)
        if not result["success"]:
            return {**result, "data": None}

        text = result["text"]
        # Strip markdown fences if present
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()

        try:
            return {**result, "data": json.loads(text)}
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "Groq returned invalid structured data. Please try again.",
                "data": None,
            }

    def translate_text(self, text: str, source_lang: str = "auto") -> Dict[str, Any]:
        """Translates non-English resume text to professional English using Groq."""
        if not self.is_configured():
            return {
                "success": False,
                "error": "Translation requires GROQ_API_KEY configured in .env or Settings.",
                "translated_text": text,
            }
        prompt = (
            "Translate the following resume document accurately into professional English. "
            "Preserve proper names, company names, job titles, and technical keywords.\n\n"
            f"SOURCE TEXT:\n\"\"\"\n{text}\n\"\"\""
        )
        result = self._generate(text, prompt, max_tokens=2500)
        return {
            "success": result["success"],
            "translated_text": result.get("text", text),
            "source_lang": source_lang,
            "error": result.get("error"),
        }
