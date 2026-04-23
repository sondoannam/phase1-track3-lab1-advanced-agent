"""
agents.py — ReAct and Reflexion agents backed by a real local LLM (Ollama).

Three module-level agent functions replace the old mock_runtime equivalents:
  - actor_answer   : calls the LLM to answer the question, optionally guided by
                     reflection_memory from previous failed attempts.
  - evaluator      : calls the LLM to judge the answer, returns a JudgeResult.
  - reflector      : calls the LLM to diagnose the failure and propose a new
                     strategy, returns a ReflectionEntry.

The BaseAgent.run() loop wires them together, accumulates real token counts
and real latency, and (for ReflexionAgent) maintains the growing
reflection_memory list passed to the Actor on every subsequent attempt.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from .llm_client import chat
from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import (
    AttemptTrace,
    JudgeResult,
    QAExample,
    ReflectionEntry,
    RunRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_context(example: QAExample) -> str:
    """Render context passages as a numbered list for the LLM."""
    parts = []
    for i, chunk in enumerate(example.context, 1):
        parts.append(f"[{i}] {chunk.title}\n{chunk.text}")
    return "\n\n".join(parts)


def _safe_json(text: str) -> dict:
    """
    Extract a JSON object from `text`, tolerating markdown fences and leading
    / trailing prose that some models emit despite instructions.
    """
    # Strip ```json ... ``` fences if present
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    # Find the first {...} block in whatever remains
    obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    if obj_match:
        text = obj_match.group(0)
    return json.loads(text)


# ---------------------------------------------------------------------------
# Agent functions
# ---------------------------------------------------------------------------

def actor_answer(
    example: QAExample,
    attempt_id: int,
    agent_type: str,
    reflection_memory: list[str],
) -> tuple[str, int, int]:
    """
    Ask the Actor LLM to answer `example.question`.

    Returns
    -------
    (answer, total_tokens, latency_ms)
    """
    context_text = _format_context(example)

    # Build the user prompt --------------------------------------------------
    reflection_block = ""
    if reflection_memory:
        numbered = "\n".join(
            f"  Attempt {i}: {entry}"
            for i, entry in enumerate(reflection_memory, 1)
        )
        reflection_block = (
            f"\n\n<reflection_memory>\n"
            f"The following lessons come from your previous failed attempt(s) "
            f"on this exact question. Apply the strategy described.\n"
            f"{numbered}\n"
            f"</reflection_memory>"
        )

    user_prompt = (
        f"Context passages:\n{context_text}"
        f"{reflection_block}"
        f"\n\nQuestion: {example.question}"
        f"\n\nAnswer (one short phrase only):"
    )

    response = chat(system_prompt=ACTOR_SYSTEM, user_prompt=user_prompt)

    # Sanitise — take only the first non-empty line to avoid padding prose
    answer = next(
        (line.strip() for line in response.content.splitlines() if line.strip()),
        response.content.strip(),
    )

    return answer, response.total_tokens, response.latency_ms


def evaluator(example: QAExample, answer: str) -> tuple[JudgeResult, int, int]:
    """
    Structured evaluator — judges `answer` against `example.gold_answer`.

    Enforcement is layered so that the first successful strategy wins and
    every subsequent one is a progressively more tolerant fallback:

    Layer 1 — Ollama native JSON mode (``format="json"``)
        The model is constrained at the grammar/token level to emit a valid
        JSON object.  This eliminates markdown fences and prose preambles
        before the text even leaves the server.

    Layer 2 — ``JudgeResult.model_validate_json()``
        Pydantic parses and validates the raw string directly.  Field
        constraints (``score`` ∈ {0, 1}, non-empty ``reason``) are enforced
        by the schema; any violation raises ``ValidationError``.

    Layer 3 — ``_safe_json()`` + manual ``JudgeResult`` construction
        If Pydantic's direct parse fails (e.g. the model produced valid JSON
        but with unexpected field names), we extract the dict with the regex
        helper and build the model field-by-field, coercing types explicitly.

    Layer 4 — Hard fallback (score=0)
        If all parsing attempts fail we return a safe default so the agent
        loop is never interrupted by an evaluator crash.

    Returns
    -------
    (JudgeResult, total_tokens, latency_ms)
    """
    context_text = _format_context(example)

    user_prompt = (
        f"Context passages:\n{context_text}"
        f"\n\nQuestion: {example.question}"
        f"\nGold answer: {example.gold_answer}"
        f"\nPredicted answer: {answer}"
        f"\n\nReturn ONLY a JSON object with keys 'score' (0 or 1) and 'reason' (string)."
    )

    # Layer 1: request Ollama JSON mode so the completion is grammar-constrained
    response = chat(
        system_prompt=EVALUATOR_SYSTEM,
        user_prompt=user_prompt,
        response_format="json",
    )
    raw: str = response.content

    # Layer 2: strict Pydantic parse directly from the JSON string
    try:
        judge = JudgeResult.model_validate_json(raw)
        return judge, response.total_tokens, response.latency_ms
    except Exception:
        pass  # fall through to Layer 3

    # Layer 3: regex extraction → dict → manual construction with coercion
    try:
        data = _safe_json(raw)
        judge = JudgeResult(
            score=int(data.get("score", 0)),
            reason=str(data.get("reason", "No reason provided.")),
        )
        return judge, response.total_tokens, response.latency_ms
    except Exception:
        pass  # fall through to Layer 4

    # Layer 4: hard fallback — log the raw output in the reason field so
    #          engineers can diagnose the failure without losing the run.
    judge = JudgeResult(
        score=0,
        reason=(
            f"[structured_evaluator fallback] All JSON parse layers failed. "
            f"Raw output (first 300 chars): {raw[:300]}"
        ),
    )
    return judge, response.total_tokens, response.latency_ms


def reflector(
    example: QAExample,
    attempt_id: int,
    judge: JudgeResult,
) -> tuple[ReflectionEntry, int, int]:
    """
    Ask the Reflector LLM to diagnose the failure and propose a new strategy.

    Returns
    -------
    (ReflectionEntry, total_tokens, latency_ms)
    """
    context_text = _format_context(example)

    user_prompt = (
        f"Context passages:\n{context_text}"
        f"\n\nQuestion: {example.question}"
        f"\nEvaluator feedback: {judge.reason}"
        f"\n\nDiagnose the failure and return ONLY valid JSON:"
    )

    response = chat(system_prompt=REFLECTOR_SYSTEM, user_prompt=user_prompt)

    # Parse JSON safely -------------------------------------------------------
    try:
        data = _safe_json(response.content)
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=str(data.get("failure_reason", judge.reason)),
            lesson=str(data.get("lesson", "Unknown lesson.")),
            next_strategy=str(data.get("next_strategy", "Try a different approach.")),
        )
    except Exception:
        # Fallback with minimal but valid ReflectionEntry
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=judge.reason,
            lesson="Could not parse reflector output; review the evaluator feedback.",
            next_strategy="Re-read every context passage carefully before answering.",
        )

    return entry, response.total_tokens, response.latency_ms


# ---------------------------------------------------------------------------
# Agent classes
# ---------------------------------------------------------------------------

@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0

        # Accumulators for real token and latency totals
        total_tokens: int = 0
        total_latency: int = 0

        for attempt_id in range(1, self.max_attempts + 1):

            # ── Actor ──────────────────────────────────────────────────────
            answer, actor_tokens, actor_latency = actor_answer(
                example, attempt_id, self.agent_type, reflection_memory
            )
            total_tokens += actor_tokens
            total_latency += actor_latency

            # ── Evaluator ──────────────────────────────────────────────────
            judge, eval_tokens, eval_latency = evaluator(example, answer)
            total_tokens += eval_tokens
            total_latency += eval_latency

            # Build trace with real numbers
            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer,
                score=judge.score,
                reason=judge.reason,
                token_estimate=actor_tokens + eval_tokens,
                latency_ms=actor_latency + eval_latency,
            )
            traces.append(trace)

            final_answer = answer
            final_score = judge.score

            # Stop early on a correct answer
            if judge.score == 1:
                break

            # ── Reflexion logic ────────────────────────────────────────────
            # Only run for reflexion agents and only when there are attempts
            # remaining (no point reflecting after the very last attempt).
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection, ref_tokens, ref_latency = reflector(
                    example, attempt_id, judge
                )
                total_tokens += ref_tokens
                total_latency += ref_latency

                # Attach reflection to the trace
                trace.reflection = reflection

                # Store in the persistent lists
                reflections.append(reflection)

                # Build a human-readable memory entry for the Actor's next turn
                memory_entry = (
                    f"[Attempt {attempt_id} failure] {reflection.failure_reason} "
                    f"Lesson: {reflection.lesson} "
                    f"Strategy for next attempt: {reflection.next_strategy}"
                )
                reflection_memory.append(memory_entry)

        # ── Determine failure mode ─────────────────────────────────────────
        failure_mode: Literal[
            "none",
            "entity_drift",
            "incomplete_multi_hop",
            "wrong_final_answer",
            "looping",
            "reflection_overfit",
        ]

        if final_score == 1:
            failure_mode = "none"
        elif self.agent_type == "reflexion" and len(reflections) >= 2:
            # Heuristic: if the agent reflected multiple times but still
            # failed it may have overfit its reasoning to the reflections.
            failure_mode = "reflection_overfit"
        else:
            failure_mode = "wrong_final_answer"

        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=bool(final_score),
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(agent_type="react", max_attempts=1)


class ReflexionAgent(BaseAgent):
    def __init__(self, max_attempts: int = 3) -> None:
        super().__init__(agent_type="reflexion", max_attempts=max_attempts)
