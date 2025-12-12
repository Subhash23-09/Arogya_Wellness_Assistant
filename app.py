import asyncio
import json
import logging

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from healthbackend.services.user_profile_store import save_profile, get_profile
from healthbackend.services.orchestrator import (
    orchestrate,
    stream_agent_updates,  # NEW: import streaming helper
)
from healthbackend.services.history_store import get_history
from healthbackend.utils.exceptions import AuthError, InputError, AgentError
from healthbackend.services.user_auth_store import check_credentials, create_user
from healthbackend.config.settings import GROQ_API_KEY, GROQ_MODEL_NAME, YOUTUBE_API_KEY
from healthbackend.services.api_key_pool import get_next_key
from healthbackend.services.youtube_recommendations import YouTubeRecommendationService
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("healthbackend")


# -------------------------
# Error Handlers
# -------------------------
@app.errorhandler(AuthError)
def auth_error(e):
    return jsonify({"error": str(e)}), 401


@app.errorhandler(InputError)
def input_error(e):
    return jsonify({"error": str(e)}), 400


# Intent Classification using LLM
def _is_health_query(text: str) -> bool:
    """LLM-based intent filter: allow only health/wellness related queries."""
    if not text or not text.strip():
        return False

    try:
        # Use LLM to determine if the query is health-related
        api_key = get_next_key()
        llm = ChatOpenAI(
            model=GROQ_MODEL_NAME,
            openai_api_base="https://api.groq.com/openai/v1",
            openai_api_key=api_key,
            temperature=0,
            max_tokens=10,
        )

        system_prompt = """You are an intent classifier. Determine if the user's query is related to health, wellness, medical symptoms, diet, fitness, mental health, or lifestyle.

Respond with ONLY "YES" if the query is health/wellness related.
Respond with ONLY "NO" if the query is completely unrelated (e.g., technology, politics, entertainment, general knowledge).

Health-related topics include: symptoms, diseases, pain, medical conditions, nutrition, diet, exercise, fitness, mental health, stress, sleep, lifestyle, medications, preventive care, body conditions (acne, skin issues, etc.)."""

        user_prompt = f"Is this query health/wellness related?\n\nQuery: {text}"

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        answer = response.content.strip().upper()

        return "YES" in answer

    except Exception as e:
        # Fallback to keyword matching if LLM fails
        t = text.lower()
        health_keywords = [
            "symptom",
            "fever",
            "cough",
            "pain",
            "headache",
            "cold",
            "flu",
            "blood pressure",
            "bp",
            "sugar",
            "diabetes",
            "hypertension",
            "cholesterol",
            "heart",
            "breath",
            "breathing",
            "asthma",
            "diet",
            "food",
            "meal",
            "nutrition",
            "calorie",
            "exercise",
            "workout",
            "walking",
            "running",
            "yoga",
            "fitness",
            "sleep",
            "insomnia",
            "snoring",
            "stress",
            "anxiety",
            "depression",
            "fatigue",
            "tired",
            "doctor",
            "medicine",
            "tablet",
            "pill",
            "health",
            "wellness",
            "weight",
            "obesity",
            "acne",
            "pimples",
            "skin",
            "rash",
            "allergy",
            "itch",
            "stomach",
            "nausea",
            "suffering",
            "feel",
            "sick",
            "unwell",
            "condition",
        ]
        return any(k in t for k in health_keywords)


# -------------------------
# Login
# -------------------------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = data.get("username", "")
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    user = check_credentials(username, password)
    if not user:
        return jsonify({"error": "Invalid username or password"}), 401

    return jsonify(
        {"status": "ok", "user_id": username, "full_name": user.get("full_name", "")}
    )


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    full_name = data.get("full_name", "").strip()

    if not username or not password or not full_name:
        return jsonify({"error": "Username, password, and full name required"}), 400

    ok = create_user(username, password, full_name)
    if not ok:
        return jsonify({"error": "Username already exists"}), 409

    return jsonify({"status": "ok", "user_id": username})


# -------------------------
# Health Assist API
# -------------------------
@app.route("/health-assist", methods=["POST"])
def health_assist():
    data = request.get_json()

    if not data or "symptoms" not in data:
        raise InputError("Symptoms required")

    symptoms = data["symptoms"].strip()
    if not _is_health_query(symptoms):
        return (
            jsonify(
                {
                    "error": "This is a wellness specialized system. Please ask about symptoms, lifestyle, diet, exercise, or other health-related topics."
                }
            ),
            400,
        )

    result = asyncio.run(
        orchestrate(
            symptoms,
            data.get("medical_report"),
            data.get("user_id", "guest"),
        )
    )
    return jsonify(result)


# -------------------------
# Recommendation API
# -------------------------
@app.route("/recommendations", methods=["POST"])
def recommendations_only():
    data = request.get_json() or {}

    symptoms = (data.get("symptoms") or "").strip()
    medical_report = data.get("medical_report", "")
    user_id = data.get("user_id", "guest")

    if not symptoms:
        raise InputError("Symptoms required")

    if not _is_health_query(symptoms):
        return (
            jsonify(
                {
                    "error": "This is a wellness specialized system. Please ask about symptoms, lifestyle, diet, exercise, or other health-related topics."
                }
            ),
            400,
        )

    result = asyncio.run(
        orchestrate(
            symptoms,
            medical_report,
            user_id,
        )
    )

    return jsonify(
        {
            "query": symptoms,
            "recommendations": result.get("recommendations", []),
        }
    )


# -------------------------
# Follow-up API (uses Groq)
# -------------------------
@app.route("/follow-up", methods=["POST"])
def follow_up():
    data = request.get_json() or {}
    user_id = data.get("user_id")
    question = data.get("question", "").strip()

    if not user_id or not question:
        return jsonify({"error": "user_id and question are required"}), 400

    history = get_history(user_id) or []
    if not history:
        return (
            jsonify({"error": "No previous wellness session found for this user"}),
            400,
        )

    last = history[-1]

    context_text = (
        f"Previous wellness plan summary:\n{last.get('synthesized_guidance', '')}\n\n"
        f"Key recommendations:\n" + "\n".join(last.get("recommendations", []))
    )

    llm = ChatOpenAI(
        model=GROQ_MODEL_NAME,
        api_key=get_next_key(),
        base_url="https://api.groq.com/openai/v1",
        temperature=0.0,
    )

    messages = [
        SystemMessage(
            content=(
                "You are a cautious wellness assistant answering follow-up questions "
                "about an existing wellness plan. Use the provided summary and "
                "recommendations as context. You may clarify, reorder, or restate "
                "information, but do NOT diagnose, do NOT prescribe medicines, and "
                "always remind the user to follow their doctor's advice."
            )
        ),
        HumanMessage(content=context_text),
        HumanMessage(content=f"User follow-up question: {question}"),
    ]

    result = llm.invoke(messages)
    return jsonify({"answer": result.content})


# -------------------------
# NEW: Streaming endpoint for DeepSeek-style UI
# -------------------------
@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    """
    Server-Sent Events endpoint.

    Streams JSON lines of the form:
      data: {"type": "thought", "content": "..."}
      data: {"type": "answer", "content": "..."}

    Frontend splits these into:
      - Thought Process panel (agent-to-agent communication)
      - Main answer bubble (final wellness plan)
    """
    data = request.get_json() or {}
    symptoms = (data.get("symptoms") or "").strip()
    medical_report = data.get("medical_report", "")

    if not symptoms:
        # SSE still needs a normal HTTP error if no symptoms
        return jsonify({"error": "Symptoms required"}), 400

    if not _is_health_query(symptoms):
        return (
            jsonify(
                {
                    "error": "This is a wellness specialized system. Please ask about symptoms, lifestyle, diet, exercise, or other health-related topics."
                }
            ),
            400,
        )

    async def agen():
        async for evt in stream_agent_updates(symptoms, medical_report):
            yield f"data: {json.dumps(evt)}\n\n"

    def generate():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async_gen = agen()
        try:
            while True:
                chunk = loop.run_until_complete(async_gen.__anext__())
                yield chunk
        except StopAsyncIteration:
            pass
        finally:
            loop.close()

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# -------------------------
# User Profile API
# -------------------------
@app.route("/profile/<user_id>", methods=["GET"])
def get_user_profile(user_id):
    return jsonify({"user_id": user_id, "profile": get_profile(user_id)})


@app.route("/profile/<user_id>", methods=["POST"])
def save_user_profile_route(user_id):
    data = request.get_json() or {}
    profile = {
        "height_cm": data.get("height_cm"),
        "weight_kg": data.get("weight_kg"),
        "medications": data.get("medications", ""),
    }
    save_profile(user_id, profile)
    return jsonify({"user_id": user_id, "profile": profile})


# -------------------------
# History API
# -------------------------
@app.route("/history/<user_id>")
def history(user_id):
    return jsonify({"user_id": user_id, "history": get_history(user_id)})


@app.route("/youtube-recommendations", methods=["POST", "OPTIONS"])
def get_youtube_recommendations():
    """
    Get YouTube video recommendations based on symptoms.

    Request body:
    {
        "symptom": "fever",
        "max_videos": 5 (optional)
    }

    Returns:
    {
        "symptom": "fever",
        "queries": [...],
        "videos": [...]
    }
    """
    # Handle CORS preflight
    if request.method == "OPTIONS":
        logger.info("Preflight /youtube-recommendations OK")
        return jsonify({"status": "ok"}), 200

    try:
        logger.info("POST /youtube-recommendations called")
        data = request.get_json()
        if not data:
            raise InputError("Request body cannot be empty")

        symptom = data.get("symptom", "").strip()
        if not symptom:
            raise InputError("Symptom field is required")

        # Enforce a hard cap of 4 videos
        requested_max = data.get("max_videos", 4)
        try:
            requested_max_int = int(requested_max)
        except Exception:
            requested_max_int = 4
        max_videos = min(max(requested_max_int, 1), 4)

        # Use YouTube API if available, otherwise fallback to mock
        if YOUTUBE_API_KEY:
            recommendations = (
                YouTubeRecommendationService.get_recommendations_with_youtube_api(
                    symptom=symptom, api_key=YOUTUBE_API_KEY, max_videos=max_videos
                )
            )
            # If API call failed, fallback to mock
            if not recommendations.get("success", False):
                recommendations = YouTubeRecommendationService.get_recommendations(
                    symptom=symptom, max_videos=max_videos
                )
        else:
            recommendations = YouTubeRecommendationService.get_recommendations(
                symptom=symptom, max_videos=max_videos
            )

        logger.info(
            "YouTube recs success=%s count=%d",
            recommendations.get("success", True),
            len(recommendations.get("videos", [])),
        )
        return jsonify(recommendations), 200

    except InputError as e:
        logger.warning("/youtube-recommendations input error: %s", e)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("/youtube-recommendations failed")
        return jsonify({"error": f"Failed to get recommendations: {str(e)}"}), 500


@app.route("/", methods=["GET"])
def welcome_health():
    return "Welcome to Health & Diet Care"


# -------------------------
# Run App
# -------------------------
if __name__ == "__main__":
    app.run(debug=False)
