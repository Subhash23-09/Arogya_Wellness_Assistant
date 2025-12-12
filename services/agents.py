from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from config.settings import GROQ_MODEL_NAME
from services.memory import get_shared_memory
from services.rag import retrieve_context
from services.api_key_pool import get_next_key, mark_key_quota_exceeded


def _make_llm_with_key():
    """
    Create a Grok chat model using the next available API key.

    Returns:
        tuple[ChatOpenAI, str]: (llm instance, api_key used)
    """
    # Pick the next key from the pool (handles round‑robin and cooldown logic)
    api_key = get_next_key()

    # Instantiate the Grok chat model with that key (OpenAI‑compatible)
    llm = ChatOpenAI(
        model=GROQ_MODEL_NAME,          # e.g. "grok-3" or "grok-3-fast"
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
async def symptom_agent(symptoms: str) -> str:
    """
    Analyze raw symptoms and comment on possible severity / urgency.
    Does NOT diagnose; only suggests when to see a doctor or seek emergency care.
    """
    memory = _memory()
    # Load previous messages so this agent can see context from other agents
    history = memory.load_memory_variables({})["chat_history"]

    # System prompt describes the role and strict safety constraints
    messages = [
        SystemMessage(
            content=(
                "You are a safe medical triage assistant. "
                "You only assess severity and suggest if the user should see a doctor. "
                "Do not provide diagnoses or prescriptions."
            )
        ),
    ] + history + [
        HumanMessage(
            content=f"Analyze these symptoms and their possible severity: {symptoms}"
        )
    ]

    llm, key = _make_llm_with_key()
    try:
        result = await llm.ainvoke(messages)
    except Exception:
        # This key may have hit quota or another hard failure → mark & retry once
        mark_key_quota_exceeded(key)
        llm, key = _make_llm_with_key()
        result = await llm.ainvoke(messages)

    memory.save_context(
        {"input": f"[symptom_agent] {symptoms}"},
        {"output": result.content},
    )
    return result.content


# ------------------------------------------------------------
# LIFESTYLE AGENT
# ------------------------------------------------------------
async def lifestyle_agent(symptoms: str) -> str:
    """
    Suggest lifestyle adjustments (sleep, stress, routine) based on symptoms
    and conversation context. Keeps suggestions generic and safe.
    """
    memory = _memory()
    history = memory.load_memory_variables({})["chat_history"]

    prompt = (
        f"Given the conversation so far and these symptoms: {symptoms}, "
        f"suggest lifestyle changes and constraints."
    )

    messages = [
        SystemMessage(
            content=(
                "You are a lifestyle coach collaborating with other agents. "
                "Suggest simple lifestyle habits, sleep hygiene, stress management, "
                "and daily routine tips. Keep suggestions safe and generic."
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
        {"input": f"[lifestyle_agent] {prompt}"},
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
        f"User symptoms: {symptoms}\n"
        f"Relevant medical report text (may be empty): {report}\n"
        f"Lifestyle information from lifestyle_agent: {lifestyle_notes}\n"
        f"Evidence / knowledge base snippets: {kb}\n\n"
        "Suggest a safe, balanced diet plan. Mention foods to prefer and foods to avoid. "
        "Highlight that this is not a replacement for a dietician or doctor."
    )

    messages = [
        SystemMessage(
            content=(
                "You are a dietician collaborating with other agents to give general diet guidance. "
                "Never claim to cure diseases or override a doctor's advice."
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
        {"input": f"[diet_agent] {prompt}"},
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
        f"User symptoms: {symptoms}\n"
        f"Diet constraints from diet_agent: {diet_notes}\n\n"
        "Recommend only low‑risk, gentle physical activities, "
        "and clearly tell the user to stop if they feel pain or discomfort."
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
        {"input": f"[fitness_agent] {prompt}"},
        {"output": result.content},
    )
    return result.content
