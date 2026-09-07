import json
import os
import random
import time
from dataclasses import dataclass

import requests

from app.agent import Agent
from app.config import config
from app.database import db

GROQ_API_KEY = config.GROQ_API_KEY
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

GROQ_GEN_MODEL = "llama-3.1-8b-instant"
GROQ_JUDGE_MODEL = "llama-3.3-70b-versatile"

def _groq_chat(model: str, system: str, user: str, temperature: float = 0.0) -> str:
    resp = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _parse_json(raw: str) -> dict:
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"  [warn] judge returned non-JSON: {raw[:150]!r}")
        return {}


# ---------- 1. fetch chunks for a document ----------

def fetch_chunks(document_id: int) -> list[dict]:
    with db.get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT chunk_index, content, page_number FROM document_chunks "
            "WHERE document_id = %s ORDER BY chunk_index",
            (document_id,),
        )
        rows = cur.fetchall()
    return [
        {"document_id": document_id, "chunk_index": r[0], "content": r[1], "page_number": r[2]}
        for r in rows
    ]


# ---------- 2. generate one question per chunk ----------

QGEN_SYSTEM = """You write one clear, natural question that is answerable using ONLY the given passage.
The question should read like something a real user would ask, not "what does the passage say about X".
Respond with JSON only: {"question": "..."}"""

def generate_question(chunk: dict) -> dict | None:
    if len(chunk["content"].strip()) < 40:
        return None  # skip near-empty chunks, not worth eval
    raw = _groq_chat(GROQ_GEN_MODEL, QGEN_SYSTEM, f"Passage:\n{chunk['content']}", temperature=0.5)
    parsed = _parse_json(raw)
    if "question" not in parsed:
        return None
    return {
        "question": parsed["question"],
        "source_document_id": chunk["document_id"],
        "source_chunk_index": chunk["chunk_index"],
        "source_content": chunk["content"],
    }



def build_eval_set(document_id: int, max_questions: int | None = None, seed: int = 42) -> list[dict]:
    chunks = fetch_chunks(document_id)
    if max_questions and len(chunks) > max_questions:
        random.Random(seed).shuffle(chunks)
        chunks = chunks[:max_questions]
    eval_set = []
    for c in chunks:
        q = generate_question(c)
        if q:
            eval_set.append(q)
    print(f"Generated {len(eval_set)} eval questions from {len(chunks)} chunks (sampled from doc)")
    return eval_set

# ---------- 3. judges ----------

FAITHFULNESS_SYSTEM = """You check whether an answer is fully supported by the given context.
Score 1 if every claim in the answer is backed by the context (or the answer correctly says
context is insufficient). Score 0 if the answer includes claims not found in the context (hallucination).
Respond with JSON only: {"score": 0 or 1, "reason": "short reason"}"""

def judge_faithfulness(question: str, answer: str, context: str) -> dict:
    user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer: {answer}"
    return _parse_json(_groq_chat(GROQ_JUDGE_MODEL, FAITHFULNESS_SYSTEM, user))


RELEVANCY_SYSTEM = """You check whether an answer actually addresses the question asked, regardless
of correctness. Score 1 if the answer is on-topic and responsive. Score 0 if it dodges,
ignores, or misses the question.
Respond with JSON only: {"score": 0 or 1, "reason": "short reason"}"""

def judge_relevancy(question: str, answer: str) -> dict:
    user = f"Question: {question}\n\nAnswer: {answer}"
    return _parse_json(_groq_chat(GROQ_JUDGE_MODEL, RELEVANCY_SYSTEM, user))

# Context
def _chunk_in_sources(source_document_id: int, source_chunk_index: int, sources: list[dict]) -> bool:
    return any(
        s.get("document_id") == source_document_id and s.get("chunk_index") == source_chunk_index
        for s in sources
    )

# ---------- 4. run eval ----------

@dataclass
class EvalResult:
    question: str
    answer: str
    faithfulness: int
    faithfulness_reason: str
    relevancy: int
    relevancy_reason: str
    context_hit: bool  # did retrieval pull back the chunk the question was generated from
    num_sources: int
    latency_ms: float


def run_eval(document_id: int, agent: Agent, max_questions: int | None = None) -> list[EvalResult]:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")

    eval_set = build_eval_set(document_id, max_questions)
    results: list[EvalResult] = []

    for i, item in enumerate(eval_set, start=1):
        print(f"[{i}/{len(eval_set)}] {item['question'][:70]}")
        start = time.time()
        outcome = agent.run(item["question"], config.TOP_K, config.TOP_N)
        latency = (time.time() - start) * 1000

        context_hit = _chunk_in_sources(
            item["source_document_id"], item["source_chunk_index"], outcome.sources
        )
        retrieved_context = "\n\n".join(s.get("full_content", s.get("content", "")) for s in outcome.sources)

        f = judge_faithfulness(item["question"], outcome.answer, retrieved_context)
        r = judge_relevancy(item["question"], outcome.answer)

        results.append(EvalResult(
            question=item["question"],
            answer=outcome.answer,
            faithfulness=f.get("score", 0),
            faithfulness_reason=f.get("reason", ""),
            relevancy=r.get("score", 0),
            relevancy_reason=r.get("reason", ""),
            context_hit=context_hit,
            num_sources=len(outcome.sources),
            latency_ms=round(latency, 1),
        ))

    return results


def summarize(results: list[EvalResult]) -> dict:
    n = len(results) or 1
    return {
        "n_questions": len(results),
        "faithfulness_avg": round(sum(r.faithfulness for r in results) / n, 3),
        "relevancy_avg": round(sum(r.relevancy for r in results) / n, 3),
        "context_recall": round(sum(r.context_hit for r in results) / n, 3),
        "avg_latency_ms": round(sum(r.latency_ms for r in results) / n, 1),
    }


def print_report(results: list[EvalResult]):
    summary = summarize(results)
    print("\n=== EVAL SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    failures = [r for r in results if r.faithfulness == 0 or not r.context_hit]
    if failures:
        print(f"\n=== {len(failures)} FAILURES ===")
        for r in failures:
            print(f"- Q: {r.question}")
            print(f"  faithful={r.faithfulness} ({r.faithfulness_reason}) | context_hit={r.context_hit}")


if __name__ == "__main__":
    from app.embedder import Embedder

    if config.LLM_PROVIDER == "groq":
        from app.llm_groq import GroqLLMClient
        llm = GroqLLMClient()
    else:
        from app.llm import LLMClient
        llm = LLMClient()

    DOC_ID = int(os.environ.get("EVAL_DOC_ID", "1"))

    embedder = Embedder()
    agent    = Agent(llm, embedder)

    results = run_eval(DOC_ID, agent, max_questions=20)
    print_report(results)

    with open("eval_results.json", "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)
    print("\nSaved eval_results.json")