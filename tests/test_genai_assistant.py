"""
Offline contract tests for the GroqCloud resume-assistant integration.
Verifies contract adherence, error sanitization, payload structures,
and grounded prompt execution using mocked API responses.
"""

from unittest.mock import MagicMock
from modules.genai_assistant import GenAIAssistant, SUPPORTED_GROQ_MODELS


def make_mock_session(response_content: str, status_code: int = 200):
    """Creates a mock requests.Session returning the specified response."""
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.json.return_value = {
        "id": "chatcmpl-test-groq-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "llama-3.3-70b-versatile",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 60, "completion_tokens": 30, "total_tokens": 90},
    }
    mock_response.text = response_content
    mock_session.post.return_value = mock_response
    return mock_session


def test_missing_key_is_safe():
    """Ensures a missing or blank API key reports an error without crashing."""
    assistant = GenAIAssistant(api_key="")
    assert assistant.is_configured() is False
    res = assistant.test_connection()
    assert res["success"] is False
    assert "GROQ_API_KEY" in assistant.configuration_message()


def test_default_model_and_endpoint():
    """Verifies default Groq configuration meets official specifications."""
    assistant = GenAIAssistant(api_key="gsk_test_mock_key")
    assert assistant.base_url == "https://api.groq.com/openai/v1"
    assert assistant.model_name in SUPPORTED_GROQ_MODELS
    assert assistant.model_name == "qwen/qwen3.8-27b"


def test_test_connection_success():
    """Verifies successful connection test call format."""
    mock_session = make_mock_session("OK")
    assistant = GenAIAssistant(api_key="gsk_valid_mock_key", session=mock_session)
    result = assistant.test_connection()

    assert result["success"] is True
    assert result["model"] == "qwen/qwen3.8-27b"

    mock_session.post.assert_called_once()
    args, kwargs = mock_session.post.call_args
    assert args[0] == "https://api.groq.com/openai/v1/chat/completions"
    assert "Bearer gsk_valid_mock_key" in kwargs["headers"]["Authorization"]
    assert kwargs["json"]["messages"][0]["content"] == "Reply with exactly: OK"


def test_dynamic_model_retrieval_filters_chat_models():
    """Verifies that get_available_models filters out audio/guard models."""
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"id": "qwen/qwen3.8-27b", "active": True},
            {"id": "whisper-large-v3", "active": True},
            {"id": "meta-llama/llama-guard-4", "active": True},
            {"id": "openai/gpt-oss-120b", "active": True},
            {"id": "inactive-model", "active": False},
        ]
    }
    mock_session.get.return_value = mock_response

    assistant = GenAIAssistant(api_key="gsk_valid_mock_key", session=mock_session)
    models = assistant.get_available_models(force_refresh=True)

    assert "qwen/qwen3.8-27b" in models
    assert "openai/gpt-oss-120b" in models
    assert "whisper-large-v3" not in models
    assert "meta-llama/llama-guard-4" not in models
    assert "inactive-model" not in models


def test_model_not_found_actionable_error():
    """Verifies that an unavailable model returns a helpful error listing accessible models."""
    mock_session = MagicMock()
    # Mock post returning 404
    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 404
    mock_post_resp.text = '{"error": {"message": "The model llama-3.3-70b-versatile does not exist or you do not have access to it.", "code": "model_not_found"}}'
    mock_session.post.return_value = mock_post_resp

    # Mock get models
    mock_get_resp = MagicMock()
    mock_get_resp.status_code = 200
    mock_get_resp.json.return_value = {
        "data": [
            {"id": "qwen/qwen3.8-27b", "active": True},
            {"id": "openai/gpt-oss-120b", "active": True},
        ]
    }
    mock_session.get.return_value = mock_get_resp

    assistant = GenAIAssistant(api_key="gsk_valid_mock_key", model_name="llama-3.3-70b-versatile", session=mock_session)
    res = assistant.test_connection()

    assert res["success"] is False
    assert "llama-3.3-70b-versatile" in res["error"]
    assert "qwen/qwen3.8-27b" in res["error"]
    assert "Settings" in res["error"]


def test_question_uses_only_uploaded_resume_context():
    """Verifies that questions are strictly grounded in uploaded resume context."""
    mock_session = make_mock_session("The candidate possesses Python and AWS experience.")
    assistant = GenAIAssistant(api_key="gsk_valid_mock_key", session=mock_session)

    result = assistant.answer_question("Skills: Python, AWS", "What skills are listed?")

    assert result["success"] is True
    assert result["answer"] == "The candidate possesses Python and AWS experience."

    args, kwargs = mock_session.post.call_args
    posted_messages = kwargs["json"]["messages"]

    # System prompt verifies evidence grounding
    assert any("evidence-grounded resume assistant" in m["content"] for m in posted_messages if m["role"] == "system")
    # User message contains UPLOADED RESUME container
    user_msg = next(m["content"] for m in posted_messages if m["role"] == "user")
    assert "UPLOADED RESUME:" in user_msg
    assert "Skills: Python, AWS" in user_msg
    assert "What skills are listed?" in user_msg


def test_generate_summary_and_improvements():
    """Verifies executive summary and improvement critique generation."""
    mock_session = make_mock_session("Candidate overview with solid backend engineering.")
    assistant = GenAIAssistant(api_key="gsk_valid_mock_key", session=mock_session)

    summary_res = assistant.generate_summary("Candidate: Mayuri Agrawal")
    assert summary_res["success"] is True
    assert "Candidate overview" in summary_res["summary"]

    critique_res = assistant.suggest_improvements("Candidate: Mayuri Agrawal")
    assert critique_res["success"] is True
    assert "Candidate overview" in critique_res["critique"]


def test_structured_insights_parse_json():
    """Verifies that structured insights request and parse JSON correctly."""
    valid_json = (
        '{"summary":"Senior engineer","skills":["Python","Docker"],"education":["B.Tech"],'
        '"experience":[],"projects":[],"improvements":[]}'
    )
    mock_session = make_mock_session(valid_json)
    assistant = GenAIAssistant(api_key="gsk_valid_mock_key", session=mock_session)

    result = assistant.generate_structured_insights("Skills: Python, Docker")

    assert result["success"] is True
    assert result["data"]["skills"] == ["Python", "Docker"]
    assert result["data"]["summary"] == "Senior engineer"

    _, kwargs = mock_session.post.call_args
    assert kwargs["json"]["response_format"] == {"type": "json_object"}


def test_error_sanitization_and_privacy():
    """Verifies that API keys and internal credentials are never leaked in error messages."""
    secret_key = "gsk_9876543210secretkey"

    # 401 Unauthorized
    msg_401 = GenAIAssistant._error_message(401, f"Unauthorized: invalid key {secret_key}")
    assert secret_key not in msg_401
    assert "authenticate" in msg_401.lower()
    assert "GROQ_API_KEY" in msg_401

    # 429 Rate limit
    msg_429 = GenAIAssistant._error_message(429, "rate_limit_exceeded: 30 requests per minute")
    assert "rate limit" in msg_429.lower()

    # Network error
    msg_net = GenAIAssistant._error_message(None, "Connection refused: https://api.groq.com")
    assert "unreachable" in msg_net.lower()
