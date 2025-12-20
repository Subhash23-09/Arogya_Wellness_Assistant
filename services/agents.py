from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from config.settings import GROQ_MODEL_NAME
from services.memory import get_shared_memory
from services.rag import retrieve_context
from services.api_key_pool import get_next_key, mark_key_quota_exceeded


def _make_llm_with_key():
    """
    Create a Groq chat model using the next available API key.

    Returns:
        tuple[ChatOpenAI, str]: (llm instance, api_key used)
    """
    api_key = get_next_key()

    llm = ChatOpenAI(
        model=GROQ_MODEL_NAME,
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.0,
    )
    return llm, api_key


def _memory():
    """
    Convenience wrapper to get the shared conversation memory object.
    All agents write into the same buffer so they can see each other’s outputs.
    """
    return get_shared_memory()


# ------------------------------------------------------------
# SYMPTOM AGENT
# ------------------------------------------------------------
async def symptom_agent(symptoms: str, report: str) -> str:
    """
    Analyze raw symptoms (and, if available, medical report text) and
    comment on possible severity / urgency.

    Does NOT diagnose; only suggests when to see a doctor or seek emergency care.
    """
    memory = _memory()
    history = memory.load_memory_variables({})["chat_history"]

    # System prompt describes the role and strict safety constraints
    messages = [
        SystemMessage(
            content=(
                "You are a safe medical triage assistant. "
                "You only assess severity and suggest if the user should see a doctor. "
                "Do not provide diagnoses or prescriptions.\n\n"
                "You may consider structured information from a lab/medical report "
                "if it is provided, but still MUST NOT diagnose or prescribe."
            )
        ),
    ] + history + [
        HumanMessage(
            content=(
                "Analyze these symptoms and their possible severity.\n\n"
                f"Symptoms:\n{symptoms}\n\n"
                f"Medical report text (may be empty):\n{report}"
            )
        )
    ]

    llm, key = _make_llm_with_key()
    try:
        result = await llm.ainvoke(messages)
    except Exception:
        mark_key_quota_exceeded(key)
        llm, key = _make_llm_with_key()
        result = await llm.ainvoke(messages)

    memory.save_context(
        {"input": f"[symptom_agent] symptoms={symptoms} report_snippet={report[:120]}"},
        {"output": result.content},
    )
    return result.content


# ------------------------------------------------------------
# LIFESTYLE AGENT
# ------------------------------------------------------------
async def lifestyle_agent(symptoms: str, report: str) -> str:
    """
    Suggest lifestyle adjustments (sleep, stress, routine) based on:
      - symptoms
      - optional medical report text
      - conversation context.

    Keeps suggestions generic and safe.
    """
    memory = _memory()
    history = memory.load_memory_variables({})["chat_history"]

    prompt = (
        "Given the conversation so far, the user's symptoms, and any available "
        "medical report text, suggest lifestyle changes and constraints.\n\n"
        f"Symptoms:\n{symptoms}\n\n"
        f"Medical report text (may be empty):\n{report}"
    )

    messages = [
        SystemMessage(
            content=(
                "You are a lifestyle coach collaborating with other agents. "
                "Suggest simple lifestyle habits, sleep hygiene, stress management, "
                "and daily routine tips. Keep suggestions safe and generic.\n\n"
                "If lab values or diagnoses are mentioned in the report, you may "
                "reference them in very general language (for example 'elevated "
                "blood sugar'), but do NOT add new diagnoses or change treatment."
            )
        ),
    ] + history + [HumanMessage(content=prompt)]

    llm, key = _make_llm_with_key()
    try:
        result = await llm.ainvoke(messages)
    except Exception:
        mark_key_quota_exceeded(key)
        llm, key = _make_llm_with_key()
        result = await llm.ainvoke(messages)

    memory.save_context(
        {"input": f"[lifestyle_agent] {prompt[:160]}"},
        {"output": result.content},
    )
    return result.content


# ------------------------------------------------------------
# DIET AGENT
# ------------------------------------------------------------
async def diet_agent(symptoms: str, report: str, lifestyle_notes: str) -> str:
    """
    Propose a safe, balanced diet plan using:
      - user symptoms
      - optional medical report text
      - output of lifestyle_agent
      - retrieved snippets from local knowledge base (RAG)

    The guidance is strictly non‑diagnostic and non‑prescriptive.
    """
    memory = _memory()
    history = memory.load_memory_variables({})["chat_history"]

    kb = retrieve_context(symptoms)

    prompt = (
        f"User symptoms:\n{symptoms}\n\n"
        f"Relevant medical report text (may be empty):\n{report}\n\n"
        f"Lifestyle information from lifestyle_agent:\n{lifestyle_notes}\n\n"
        f"Evidence / knowledge base snippets:\n{kb}\n\n"
        "Suggest a safe, balanced diet plan. Mention foods to prefer and foods to avoid. "
        "Highlight that this is not a replacement for a dietician or doctor."
    )

    messages = [
        SystemMessage(
            content=(
                "You are a dietician collaborating with other agents to give general diet guidance. "
                "Never claim to cure diseases or override a doctor's advice. "
                "Do not name prescription medicines or doses."
            )
        ),
    ] + history + [HumanMessage(content=prompt)]

    llm, key = _make_llm_with_key()
    try:
        result = await llm.ainvoke(messages)
    except Exception:
        mark_key_quota_exceeded(key)
        llm, key = _make_llm_with_key()
        result = await llm.ainvoke(messages)

    memory.save_context(
        {"input": f"[diet_agent] {prompt[:160]}"},
        {"output": result.content},
    )
    return result.content


# ------------------------------------------------------------
# FITNESS AGENT
# ------------------------------------------------------------
async def fitness_agent(symptoms: str, diet_notes: str) -> str:
    """
    Recommend gentle, low‑risk physical activities that respect
    both the symptoms and the diet constraints.
    Always reminds the user to stop if they feel discomfort and to
    consult a doctor before more intense exercise.
    """
    memory = _memory()
    history = memory.load_memory_variables({})["chat_history"]

    prompt = (
        f"User symptoms:\n{symptoms}\n\n"
        f"Diet constraints from diet_agent:\n{diet_notes}\n\n"
        "Recommend only low‑risk, gentle physical activities, and clearly tell the "
        "user to stop if they feel pain or discomfort. Always remind them to talk "
        "to their doctor before starting or intensifying exercise."
    )

    messages = [
        SystemMessage(
            content=(
                "You are a cautious fitness coach. "
                "You design simple, low‑intensity plans that are generally safe. "
                "Always recommend consulting a doctor before heavy exercise."
            )
        ),
    ] + history + [HumanMessage(content=prompt)]

    llm, key = _make_llm_with_key()
    try:
        result = await llm.ainvoke(messages)
    except Exception:
        mark_key_quota_exceeded(key)
        llm, key = _make_llm_with_key()
        result = await llm.ainvoke(messages)

    memory.save_context(
        {"input": f"[fitness_agent] {prompt[:160]}"},
        {"output": result.content},
    )
    return result.content
