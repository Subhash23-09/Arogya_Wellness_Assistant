import json
import re
from typing import AsyncGenerator, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from healthbackend.services.api_key_pool import (
    get_next_key,
    mark_key_quota_exceeded,
)
from healthbackend.services.agents import (
    symptom_agent,
    lifestyle_agent,
    diet_agent,
    fitness_agent,
)
from healthbackend.services.history_store import save_history
from healthbackend.services.memory import get_shared_memory, reset_memory
from healthbackend.config.settings import GROQ_MODEL_NAME


# ---------------------------------------------------------------------
# Synthesizer LLM helper
# ---------------------------------------------------------------------
def _make_synth_llm_with_key():
    """Create the synthesizer LLM using the next available Groq API key."""
    key = get_next_key()
    llm = ChatOpenAI(
        model=GROQ_MODEL_NAME,  # e.g. "llama-3.3-70b-versatile"
        api_key=key,
        base_url="https://api.groq.com/openai/v1",
        temperature=0.0,  # deterministic JSON
    )
    return llm, key


def _build_markdown_table(output: dict) -> str:
    """Build a markdown-style block summarizing each agent."""
    parts: list[str] = []

    def add_block(title: str, text: str | None):
        if not text:
            return
        if parts:
            parts.append("")
        parts.append(f"**{title}**")
        parts.append(text.strip())

    add_block("Symptom agent", output.get("symptom_analysis", ""))
    add_block("Lifestyle agent", output.get("lifestyle", ""))
    add_block("Diet agent", output.get("diet", ""))
    add_block("Fitness agent", output.get("fitness", ""))

    return "\n".join(parts)


# ---------------------------------------------------------------------
# Main orchestration (non‑streaming, used by /health-assist)
# ---------------------------------------------------------------------
async def orchestrate(
    symptoms: str, medical_report: str, user_id: str
) -> Dict[str, Any]:
    """Run the full multi‑agent pipeline and return structured JSON."""

    # Reset shared memory for this session
    reset_memory()
    memory = get_shared_memory()

    # Track agent communication flow
    agent_flow = []

    # 1. Symptom agent
    symptom_result = await symptom_agent(symptoms)
    agent_flow.append(
        {
            "agent": "Symptom Agent",
            "output": symptom_result,
        }
    )

    # 2. Lifestyle agent
    lifestyle_result = await lifestyle_agent(symptoms)
    agent_flow.append(
        {
            "agent": "Lifestyle Agent",
            "output": lifestyle_result,
        }
    )

    # 3. Diet agent (uses lifestyle)
    diet_result = await diet_agent(
        symptoms=symptoms,
        report=medical_report,
        lifestyle_notes=lifestyle_result,
    )
    agent_flow.append(
        {
            "agent": "Diet Agent",
            "output": diet_result,
        }
    )

    # 4. Fitness agent (uses diet)
    fitness_result = await fitness_agent(
        symptoms=symptoms,
        diet_notes=diet_result,
    )
    agent_flow.append(
        {
            "agent": "Fitness Agent",
            "output": fitness_result,
        }
    )

    # Conversation history for synthesis
    history = memory.load_memory_variables({})["chat_history"]

    # -----------------------------------------------------------------
    # Synthesizer LLM: combine all agent outputs into JSON wellness plan
    # -----------------------------------------------------------------
    synth_llm, synth_key = _make_synth_llm_with_key()

    synth_messages = [
        SystemMessage(
            content=(
                "You are an orchestrator summarizing a mild to moderate health concern.\n"
                "Read the full conversation between symptom_agent, lifestyle_agent, "
                "diet_agent, and fitness_agent.\n\n"
                "Write a concise, well-structured wellness plan in markdown with these sections:\n"
                "1. Overview – 2-3 sentences summarizing the situation and overall goal.\n"
                "2. When to See a Doctor – 2-4 bullet points, clearly describing red-flag symptoms.\n"
                "3. Lifestyle & Rest – 3-5 bullet points with specific, gentle daily actions.\n"
                "4. Hydration & Diet – 3-5 bullet points with simple, safe food and fluid guidance.\n"
                "5. Hygiene & Environment – 2-4 bullet points to reduce irritation and infection spread.\n"
                "6. Movement & Activity – 2-4 bullet points with ONLY low-intensity options, "
                "including a bold STOP warning for chest pain, breathing difficulty, dizziness, "
                "or marked worsening.\n"
                "7. Final Note – 1-2 sentences reminding that this is not a diagnosis and to "
                "follow a doctor's advice.\n\n"
                "Tone: calm, reassuring, non-alarming, strictly non-diagnostic. "
                "Never name specific medicines or doses. Never say you replace a doctor.\n\n"
                "Return ONLY valid JSON with keys:\n"
                "  - synthesized_guidance: the markdown text described above\n"
                "  - recommendations: array of short, plain-language recommendation strings\n"
                "Do not wrap JSON in code fences or add any extra text."
            )
        ),
        *history,
        HumanMessage(content="Generate the JSON response now."),
    ]

    try:
        final_answer = await synth_llm.ainvoke(synth_messages)
    except Exception:
        # Key likely hit quota or hard failure → rotate and retry once
        mark_key_quota_exceeded(synth_key)
        synth_llm, synth_key = _make_synth_llm_with_key()
        final_answer = await synth_llm.ainvoke(synth_messages)

    raw = final_answer.content.strip()

    # ------------------------------------------------------------
    # JSON Cleaning and Parsing
    # ------------------------------------------------------------
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"synthesized_guidance": raw, "recommendations": []}

    # ------------------------------------------------------------
    # Final Output Structure
    # ------------------------------------------------------------
    output = {
        "user_id": user_id,
        "query": symptoms,
        "symptom_analysis": symptom_result,
        "lifestyle": lifestyle_result,
        "diet": diet_result,
        "fitness": fitness_result,
        "synthesized_guidance": data.get("synthesized_guidance", ""),
        "recommendations": data.get("recommendations", []),
        "agent_flow": agent_flow,
    }

    # Add markdown table summary
    output["table_markdown"] = _build_markdown_table(output)

    save_history(user_id, output)
    return output


# ---------------------------------------------------------------------
# Streaming helper - agent communication for UI
# ---------------------------------------------------------------------
async def stream_agent_updates(symptoms: str, medical_report: str) -> AsyncGenerator[Dict[str, str], None]:
    """
    Async generator that yields 'thought' and 'answer' events for the UI,
    reflecting a mesh-style multi-agent workflow:

    User -> Orchestrator -> Symptom
    Symptom -> Diet + Lifestyle
    Lifestyle <-> Diet, Diet -> Fitness, Fitness -> Lifestyle
    All -> Synthesizer -> Orchestrator -> User
    """

    reset_memory()
    memory = get_shared_memory()

    # 1) Symptom Agent
    yield {
        "type": "thought",
        "content": "User → Orchestrator: new wellness query received.",
    }
    yield {
        "type": "thought",
        "content": "Orchestrator → SymptomAgent: analyze primary symptoms.",
    }
    symptom_result = await symptom_agent(symptoms)
    yield {
        "type": "thought",
        "content": (
            "SymptomAgent → Orchestrator: symptom profile ready "
            f"(example: {symptom_result[:160]}...)."
        ),
    }

    # 2) Symptom → Diet + Lifestyle (parallel conceptually)
    yield {
        "type": "thought",
        "content": "SymptomAgent → DietAgent: sending symptom profile for diet constraints.",
    }
    yield {
        "type": "thought",
        "content": "SymptomAgent → LifestyleAgent: sending symptom profile for lifestyle checks.",
    }

    # 3) Lifestyle Agent – first pass
    lifestyle_result = await lifestyle_agent(symptoms)
    yield {
        "type": "thought",
        "content": (
            "LifestyleAgent → Orchestrator: first-pass lifestyle guidance ready "
            f"(sleep, routine, stress). Example: {lifestyle_result[:160]}..."
        ),
    }

    # 4) Diet Agent – uses symptoms + lifestyle constraints
    yield {
        "type": "thought",
        "content": "Orchestrator → DietAgent: generate plan using symptoms + lifestyle constraints.",
    }
    diet_result = await diet_agent(
        symptoms=symptoms,
        report=medical_report,
        lifestyle_notes=lifestyle_result,
    )
    yield {
        "type": "thought",
        "content": (
            "DietAgent → Orchestrator: diet & hydration plan ready. "
            f"Example: {diet_result[:160]}..."
        ),
    }

    # Diet → Fitness
    yield {
        "type": "thought",
        "content": (
            "DietAgent → FitnessAgent: sending energy & restriction profile "
            "to shape safe activity level."
        ),
    }

    # 5) Fitness Agent – uses diet restrictions
    fitness_result = await fitness_agent(
        symptoms=symptoms,
        diet_notes=diet_result,
    )
    yield {
        "type": "thought",
        "content": (
            "FitnessAgent → Orchestrator: movement plan ready "
            f"(light / restricted). Example: {fitness_result[:160]}..."
        ),
    }

    # Fitness → Lifestyle: check for behavior conflicts
    yield {
        "type": "thought",
        "content": (
            "FitnessAgent → LifestyleAgent: sharing activity plan to detect "
            "conflicts with fatigue, sleep, or routine."
        ),
    }

    # 6) Lifestyle Agent – second pass, refines using fitness & diet
    refined_lifestyle_prompt = (
        f"Symptoms: {symptoms}\n\n"
        f"Diet plan summary:\n{diet_result}\n\n"
        f"Fitness plan summary:\n{fitness_result}\n\n"
        "Adjust lifestyle guidance if any conflicts or overloads are detected."
    )
    refined_lifestyle = await lifestyle_agent(refined_lifestyle_prompt)
    yield {
        "type": "thought",
        "content": (
            "LifestyleAgent → DietAgent & FitnessAgent: updated lifestyle "
            "constraints (sleep, stress, routine) shared for consistency."
        ),
    }
    yield {
        "type": "thought",
        "content": (
            "LifestyleAgent → Orchestrator: refined lifestyle guidance ready. "
            f"Example: {refined_lifestyle[:160]}..."
        ),
    }

    # 7) All agents → Synthesizer
    history = memory.load_memory_variables({})["chat_history"]
    synth_llm, synth_key = _make_synth_llm_with_key()

    yield {
        "type": "thought",
        "content": (
            "Orchestrator → OutputSynthesizer: combining Symptom, Diet, "
            "Lifestyle, and Fitness outputs into one plan."
        ),
    }

    synth_messages = [
        SystemMessage(
            content=(
                "You are an orchestrator summarizing a mild to moderate health concern.\n"
                "You receive outputs from:\n"
                "- SymptomAgent (symptom profile)\n"
                "- LifestyleAgent (may be called twice: initial + refined)\n"
                "- DietAgent (diet & hydration plan)\n"
                "- FitnessAgent (movement plan)\n\n"
                "Produce a single safe wellness plan in markdown. "
                "Do not diagnose or prescribe medicines."
            )
        ),
        *history,
        HumanMessage(content="Generate the wellness plan now."),
    ]

    try:
        async for chunk in synth_llm.astream(synth_messages):
            text = getattr(chunk, "content", "") or ""
            if text:
                # Stream final answer tokens
                yield {"type": "answer", "content": text}
    except Exception:
        mark_key_quota_exceeded(synth_key)
        synth_llm, synth_key = _make_synth_llm_with_key()
        async for chunk in synth_llm.astream(synth_messages):
            text = getattr(chunk, "content", "") or ""
            if text:
                yield {"type": "answer", "content": text}

    # 8) Final delivery
    yield {
        "type": "thought",
        "content": "OutputSynthesizer → Orchestrator → User: final wellness plan delivered.",
    }