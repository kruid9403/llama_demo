import os
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from arxiv_ingest import ingest_query
from crossref_ingest import ingest_crossref_query
from llama_demo import _compute_eot_ids, load_model
from vector_store.retrieval import retrieve_chunks
from url_loader import load_url_chunks

load_dotenv()

STATUS_PREFIX = "[[STATUS]]"
MAX_TURNS = int(os.getenv("DEBATE_MAX_TURNS", "8"))
ARXIV_RESULTS = int(os.getenv("DEBATE_ARXIV_RESULTS", os.getenv("ARXIV_MAX_RESULTS", "6")))
CROSSREF_RESULTS = int(os.getenv("DEBATE_CROSSREF_RESULTS", "6"))
MAX_NEW_TOKENS = int(os.getenv("DEBATE_MAX_NEW_TOKENS", "1000"))

model, tokenizer = load_model()


class InterruptStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_event: threading.Event, local_stop_event: threading.Event | None = None):
        super().__init__()
        self.stop_event = stop_event
        self.local_stop_event = local_stop_event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.stop_event.is_set() or (self.local_stop_event is not None and self.local_stop_event.is_set())


@dataclass
class ScientistTurn:
    scientist: str
    query: str
    opinion: str
    code: str
    response: str
    agreement: str
    viability: str
    validation: str
    conclusion: str


def _emit_status(stream_queue: Any, message: str) -> None:
    if stream_queue is not None:
        stream_queue.put(f"\n{STATUS_PREFIX} {message}\n")


def _emit_text(stream_queue: Any, text: str) -> None:
    if stream_queue is not None:
        stream_queue.put(text)


def _build_query(question: str, last_signal: str) -> str:
    seed = f"{question}\n{last_signal}".strip().lower()
    seed = re.sub(r"\[[0-9]+\]", " ", seed)
    seed = re.sub(r"[^a-z0-9\s]", " ", seed)
    seed = re.sub(r"\s+", " ", seed).strip()
    tokens = [t for t in seed.split() if len(t) > 2 and t not in {"title", "source", "abstract", "venue"}]
    return " ".join(tokens[:22]) or question


def _parse_signal(label: str, text: str, default: str) -> str:
    pattern = rf"^{label}:\s*(.+)$"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return (match.group(1).strip() if match else default)


def _normalize_agreement(raw: str) -> str:
    value = (raw or "").strip().lower()
    if "disagree" in value:
        return "DISAGREE"
    if "agree" in value:
        return "AGREE"
    return "DISAGREE"


def _norm_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2}


def _normalize_viability(raw: str) -> str:
    value = (raw or "").strip().lower()
    if "viable" in value and "not" not in value:
        return "VIABLE"
    return "NOT_VIABLE"


def _normalize_validation(raw: str) -> str:
    value = (raw or "").strip().lower()
    if "pass" in value and "fail" not in value:
        return "PASS"
    if "n/a" in value or "na" == value:
        return "N/A"
    return "FAIL"


def _normalize_answer_verdict(raw: str) -> str:
    value = (raw or "").strip().lower()
    if "verified" in value and "unverified" not in value:
        return "VERIFIED"
    if "unverified" in value:
        return "UNVERIFIED"
    return "UNVERIFIED"


def _extract_first_code_block(text: str) -> str:
    match = re.search(r"```(?:python)?\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _build_prompt(
    scientist: str,
    question: str,
    prior_turns: List[ScientistTurn],
    retrieved: List[Dict[str, Any]],
    opinion_to_verify: str,
) -> str:
    stance = (
        "You are deliberately optimistic: propose feasible paths, practical improvements, and near/mid-term opportunities."
        if scientist == "Research Scientist 1"
        else "You are deliberately pessimistic: stress constraints, failure modes, scaling bottlenecks, and overclaim risk."
    )
    prior = "\n\n".join(
        f"{t.scientist} prior response:\n{t.response}\nAgreement: {t.agreement}\nConclusion: {t.conclusion}"
        for t in prior_turns[-2:]
    )
    context = "\n\n".join(
        (
            f"[{idx+1}] {c.get('content', '')}\n"
            f"Source: {c.get('document', 'unknown')} | {c.get('section', 'section')} | {c.get('url', 'unknown')}"
        )
        for idx, c in enumerate(retrieved)
    )

    role_block = (
        """
Role-specific output requirements:
- You are Scientist 1 (solution proposer).
- Provide a concrete solution attempt with runnable Python code.
- Include one fenced Python block.
- Set `Validation: N/A`.
Required body:
Proposed Approach: <2-4 sentences>
Proposed Code:
```python
<code>
```
Why This Should Work: <2-4 sentences with citations>
Next Test: <1 sentence>
References (APA 7):
[R1] <APA 7 reference>
[R2] <APA 7 reference>
"""
        if scientist == "Research Scientist 1"
        else """
Role-specific output requirements:
- You are Scientist 2 (code validator).
- Validate the proposed code from Scientist 1 using evidence and implementation realism.
- Decide if it is correct/viable at the requested scale.
- If invalid, provide a corrected minimal code patch in one fenced Python block.
Required body:
Code Review Verdict: <1-2 sentences>
Validation Findings:
1. <finding>
2. <finding>
Corrected Code (only if needed):
```python
<optional patch>
```
Next Test: <1 sentence>
References (APA 7):
[R1] <APA 7 reference>
[R2] <APA 7 reference>
Final verification ending (exactly once, at the end):
Answer: VERIFIED or UNVERIFIED
Verification Reason: <1-2 sentences>
"""
    )

    return f"""
You are {scientist}, a strict research scientist.
{stance}

Rules:
- Use only peer-reviewed or scholarly sources from context.
- Do not use Wikipedia or unsourced claims.
- Critique weak evidence from prior turn before refining the answer.
- Keep answer technical and argumentative.
- Use in-text citations and provide full references in APA 7 format.
- Use inline citation keys in the form `[R1]`, `[R2]`, etc. directly in response text.
- Only include references that are cited inline with `[R#]`.
- Every reference in `References (APA 7)` must have at least one matching inline `[R#]` citation.
- Do not include uncited references.
- Do not output a bare list of references.
- Do not copy source snippets verbatim; synthesize and reason from evidence.
- Every major claim must map to retrieved evidence.
- Response length is flexible; do not exceed 1000 tokens.
- Output exactly one response block and then stop.
- Do not repeat headers, sections, or the Agreement/Conclusion pair.
- Do not include meta text like "Machine generated answer" or "Note:".
- Do not prefix with "My response:".
- Treat the prior opinion as a hypothesis to verify or refute with evidence.
- Do not emit extra closings like "End of Machine Response", "THE END", or repeated "Final Answer".
- Include the three required machine lines exactly once:
Agreement: AGREE or DISAGREE
Viability: VIABLE or NOT_VIABLE
Validation: PASS or FAIL or N/A
Conclusion: <single sentence final claim>
- The final section of the output must be `References (APA 7)` at the end.

User question:
{question}

Opinion To Verify:
{opinion_to_verify}

Recent debate context:
{prior if prior else 'No prior turns.'}

Retrieved scholarly evidence:
{context if context else 'No retrieved context.'}

{role_block}
""".strip()


def _should_early_stop(output_text: str) -> bool:
    # If model starts emitting a second structured block, cut generation.
    if output_text.count("\nAgreement:") > 1:
        return True
    if output_text.count("\nConclusion:") > 1:
        return True
    if output_text.lower().count("machine generated answer") > 0:
        return True
    if output_text.count("\nPosition:") > 1:
        return True
    if output_text.count("\nViability:") > 1:
        return True
    if output_text.count("\nValidation:") > 1:
        return True
    if output_text.count("\nReferences (APA 7):") > 1:
        return True
    lower = output_text.lower()
    boilerplate_markers = [
        "end of machine response",
        "the end.",
        "final answer.",
        "answer verified.",
        "verification complete.",
    ]
    if any(marker in lower for marker in boilerplate_markers):
        return True
    return False


def _generate_streamed_text(prompt: str, stream_queue: Any, stop_event: threading.Event) -> str:
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    eos_ids = _compute_eot_ids(tokenizer) or [tokenizer.eos_token_id]
    local_stop_event = threading.Event()

    errors: List[Exception] = []

    def _run_generation() -> None:
        try:
            model.generate(
                inputs=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                streamer=streamer,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.12,
                no_repeat_ngram_size=6,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([InterruptStoppingCriteria(stop_event, local_stop_event)]),
            )
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    t = threading.Thread(target=_run_generation, daemon=True)
    t.start()

    out = ""
    for token in streamer:
        if stop_event.is_set():
            break
        out += token
        _emit_text(stream_queue, token)
        if _should_early_stop(out):
            local_stop_event.set()
            break

    t.join(timeout=2)
    if errors:
        raise errors[0]
    return out.strip()


def run_research_debate(
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    stream_queue: Any = None,
    stop_event: Optional[threading.Event] = None,
    url: Optional[str] = None,
) -> str:
    stop_event = stop_event or threading.Event()
    history = history or []
    turns: List[ScientistTurn] = []

    _emit_status(stream_queue, "Starting two-scientist proposal-review cycle...")
    _emit_text(stream_queue, "\n=== Two-Scientist Debate Started ===\n")

    # One cycle = Scientist 1 proposal + Scientist 2 review
    for cycle_idx in range(MAX_TURNS):
        if stop_event.is_set():
            _emit_status(stream_queue, "Interrupted by user.")
            break

        scientist = "Research Scientist 1"
        last_signal = turns[-1].conclusion if turns else ""
        opinion_to_verify = (
            turns[-1].opinion
            if turns
            else f"Initial working opinion: {question}"
        )
        query = _build_query(question, last_signal)

        _emit_status(stream_queue, f"{scientist}: searching arXiv and scholarly indexes...")
        _emit_text(stream_queue, f"\n{scientist} Search Query: {query}\n")
        _emit_text(stream_queue, f"{scientist} Opinion To Verify: {opinion_to_verify}\n")

        arxiv_count = 0
        crossref_count = 0
        try:
            arxiv_count = ingest_query(query, max_results=ARXIV_RESULTS)
        except Exception as exc:
            _emit_text(stream_queue, f"{scientist} arXiv ingest error: {exc}\n")

        if stop_event.is_set():
            _emit_status(stream_queue, "Interrupted by user.")
            break

        try:
            crossref_count = ingest_crossref_query(query, max_results=CROSSREF_RESULTS)
        except Exception as exc:
            _emit_text(stream_queue, f"{scientist} Crossref ingest error: {exc}\n")

        if stop_event.is_set():
            _emit_status(stream_queue, "Interrupted by user.")
            break

        _emit_text(
            stream_queue,
            f"{scientist} Ingested: arXiv={arxiv_count}, Crossref={crossref_count}\n",
        )

        _emit_status(stream_queue, f"{scientist}: retrieving evidence...")
        retrieved = retrieve_chunks(question, top_k=12, recent_only=False)
        _emit_text(stream_queue, f"{scientist} Retrieved chunks from vector DB: {len(retrieved)}\n")
        if url:
            try:
                retrieved = load_url_chunks(url) + retrieved
            except Exception as exc:
                _emit_text(stream_queue, f"Optional URL load error: {exc}\n")

        _emit_text(stream_queue, f"\n{scientist} Response:\n")
        _emit_status(stream_queue, f"{scientist}: generating response...")
        prompt = _build_prompt(scientist, question, turns, retrieved, opinion_to_verify)
        response = _generate_streamed_text(prompt, stream_queue, stop_event)
        _emit_text(stream_queue, "\n")

        agreement = _normalize_agreement(_parse_signal("Agreement", response, "DISAGREE"))
        conclusion = _parse_signal("Conclusion", response, "No conclusion provided")

        current_turn = ScientistTurn(
            scientist=scientist,
            query=query,
            opinion=conclusion,
            code=_extract_first_code_block(response),
            response=response,
            agreement=agreement,
            viability=_normalize_viability(_parse_signal("Viability", response, "NOT_VIABLE")),
            validation=_normalize_validation(_parse_signal("Validation", response, "N/A")),
            conclusion=conclusion,
        )
        turns.append(current_turn)

        if stop_event.is_set():
            break

        scientist = "Research Scientist 2"
        last_signal = turns[-1].conclusion
        code_to_verify = turns[-1].code
        opinion_to_verify = f"{turns[-1].opinion}\n\nCode To Validate:\n```python\n{code_to_verify}\n```" if code_to_verify else turns[-1].opinion
        query = _build_query(question, last_signal)

        _emit_status(stream_queue, f"{scientist}: reviewing viability...")
        _emit_text(stream_queue, f"\n{scientist} Search Query: {query}\n")
        _emit_text(stream_queue, f"{scientist} Opinion To Verify: {opinion_to_verify}\n")

        arxiv_count = 0
        crossref_count = 0
        try:
            arxiv_count = ingest_query(query, max_results=ARXIV_RESULTS)
        except Exception as exc:
            _emit_text(stream_queue, f"{scientist} arXiv ingest error: {exc}\n")

        if stop_event.is_set():
            _emit_status(stream_queue, "Interrupted by user.")
            break

        try:
            crossref_count = ingest_crossref_query(query, max_results=CROSSREF_RESULTS)
        except Exception as exc:
            _emit_text(stream_queue, f"{scientist} Crossref ingest error: {exc}\n")

        if stop_event.is_set():
            _emit_status(stream_queue, "Interrupted by user.")
            break

        _emit_text(
            stream_queue,
            f"{scientist} Ingested: arXiv={arxiv_count}, Crossref={crossref_count}\n",
        )

        _emit_status(stream_queue, f"{scientist}: retrieving evidence...")
        retrieved = retrieve_chunks(question, top_k=12, recent_only=False)
        _emit_text(stream_queue, f"{scientist} Retrieved chunks from vector DB: {len(retrieved)}\n")
        if url:
            try:
                retrieved = load_url_chunks(url) + retrieved
            except Exception as exc:
                _emit_text(stream_queue, f"Optional URL load error: {exc}\n")

        _emit_text(stream_queue, f"\n{scientist} Response:\n")
        _emit_status(stream_queue, f"{scientist}: generating response...")
        prompt = _build_prompt(scientist, question, turns, retrieved, opinion_to_verify)
        response = _generate_streamed_text(prompt, stream_queue, stop_event)
        _emit_text(stream_queue, "\n")

        agreement = _normalize_agreement(_parse_signal("Agreement", response, "DISAGREE"))
        viability = _normalize_viability(_parse_signal("Viability", response, "NOT_VIABLE"))
        conclusion = _parse_signal("Conclusion", response, "No conclusion provided")
        answer_verdict = _normalize_answer_verdict(_parse_signal("Answer", response, "UNVERIFIED"))
        validation = _normalize_validation(_parse_signal("Validation", response, "FAIL"))
        if answer_verdict == "VERIFIED":
            validation = "PASS"
            viability = "VIABLE"
        elif answer_verdict == "UNVERIFIED":
            validation = "FAIL"
            viability = "NOT_VIABLE"

        reviewer_turn = ScientistTurn(
            scientist=scientist,
            query=query,
            opinion=conclusion,
            code=_extract_first_code_block(response),
            response=response,
            agreement=agreement,
            viability=viability,
            validation=validation,
            conclusion=conclusion,
        )
        turns.append(reviewer_turn)

        if reviewer_turn.viability == "VIABLE" and reviewer_turn.validation == "PASS":
            _emit_status(stream_queue, "Scientist 2 marked proposal viable.")
            _emit_text(stream_queue, "\n=== Viable Solution Reached ===\n")
            _emit_text(stream_queue, f"Final Conclusion: {reviewer_turn.conclusion}\n")
            break
        _emit_status(stream_queue, "Scientist 2 marked proposal NOT_VIABLE/FAIL. Scientist 1 will revise.")

    if stop_event.is_set():
        _emit_text(stream_queue, "\n=== Debate Interrupted ===\n")
    elif len(turns) >= MAX_TURNS * 2 and turns and turns[-1].viability != "VIABLE":
        _emit_status(stream_queue, "Max cycles reached without viable approval.")
        _emit_text(stream_queue, "\n=== Max Cycles Reached (Not Yet Viable) ===\n")
        if turns:
            _emit_text(stream_queue, f"Latest Conclusion: {turns[-1].conclusion}\n")

    transcript_parts = []
    for t in turns:
        transcript_parts.append(f"{t.scientist} Query: {t.query}\n{t.response}")
    return "\n\n".join(transcript_parts)
