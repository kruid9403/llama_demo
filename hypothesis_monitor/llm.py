import os
import threading
from datetime import datetime, timezone
from typing import Any

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from hypothesis_monitor.engine import build_llama_hypothesis_prompt, is_contrived_hypothesis

load_dotenv()

MODEL_ID = os.getenv("HYPOTHESIS_MODEL_ID", os.getenv("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct"))
TOKENIZER_ID = os.getenv("HYPOTHESIS_TOKENIZER_ID", os.getenv("TOKENIZER_ID", MODEL_ID))
USE_8BIT = os.getenv("HYPOTHESIS_USE_8BIT", os.getenv("USE_8BIT", "1")) == "1"
DEVICE_HINT = "cuda" if os.getenv("USE_CUDA", "1") == "1" else "cpu"

MAX_NEW_TOKENS = int(os.getenv("HYPOTHESIS_MAX_NEW_TOKENS", "420"))
MAX_INPUT_TOKENS = int(os.getenv("HYPOTHESIS_MAX_INPUT_TOKENS", "6000"))
TEMPERATURE = float(os.getenv("HYPOTHESIS_TEMPERATURE", "0.9"))
TOP_P = float(os.getenv("HYPOTHESIS_TOP_P", "0.92"))
REPETITION_PENALTY = float(os.getenv("HYPOTHESIS_REPETITION_PENALTY", "1.08"))
NOVELTY_REWRITE_ATTEMPTS = max(0, int(os.getenv("HYPOTHESIS_NOVELTY_REWRITE_ATTEMPTS", "1")))

_MODEL = None
_TOKENIZER = None
_MODEL_LOCK = threading.Lock()
_GEN_LOCK = threading.Lock()


class HypothesisNoveltyError(RuntimeError):
    pass


def _compute_eot_ids(tokenizer) -> list[int]:
    eot_ids = [tokenizer.eos_token_id]
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    except Exception:
        eot_id = None
    if eot_id is not None and eot_id != tokenizer.eos_token_id:
        eot_ids.append(eot_id)
    return eot_ids


def get_model_id() -> str:
    return MODEL_ID


def _load_model():
    global _MODEL, _TOKENIZER
    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER

    with _MODEL_LOCK:
        if _MODEL is not None and _TOKENIZER is not None:
            return _MODEL, _TOKENIZER

        hf_token = os.getenv("HF_TOKEN", "").strip() or None

        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_ID,
            token=hf_token,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        model_kwargs = dict(
            device_map="auto",
            token=hf_token,
            low_cpu_mem_usage=True,
            offload_buffers=True,
        )
        if USE_8BIT:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            model_kwargs["torch_dtype"] = torch.float16 if DEVICE_HINT == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
        model.eval()

        _MODEL = model
        _TOKENIZER = tokenizer
        return _MODEL, _TOKENIZER


def _build_chat_input(tokenizer, prompt: str):
    messages = [
        {"role": "system", "content": "You are a research scientist and hypothesis-generation assistant."},
        {"role": "user", "content": prompt},
    ]
    try:
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        chat_text = (
            "System: You are a research scientist and hypothesis-generation assistant.\n\n"
            f"User:\n{prompt}\n\nAssistant:"
        )
    enc = tokenizer(
        chat_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
    )
    return enc


def _hypothesis_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    return f"HYP-{stamp}"


def _generate_text(model, tokenizer, prompt: str) -> tuple[str, int]:
    enc = _build_chat_input(tokenizer, prompt)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    eos_ids = _compute_eot_ids(tokenizer) or [tokenizer.eos_token_id]

    with _GEN_LOCK:
        with torch.inference_mode():
            out = model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                repetition_penalty=REPETITION_PENALTY,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
            )

    generated_ids = out[0][input_ids.shape[1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text, int(input_ids.shape[1])


def _novelty_retry_prompt(base_prompt: str, rejected_text: str) -> str:
    rejected_excerpt = (rejected_text or "").strip()
    if len(rejected_excerpt) > 1400:
        rejected_excerpt = rejected_excerpt[:1400] + "..."
    return (
        f"{base_prompt}\n\n"
        "Validation feedback:\n"
        "The previous draft was rejected because it read like a stitched combination of existing theories "
        "(combine/integrate/hybrid/fusion framing).\n"
        "Rewrite from scratch with a single original mechanism and no composite framing language.\n\n"
        "Rejected draft excerpt:\n"
        f"{rejected_excerpt}\n"
    )


def generate_hypothesis_from_context(
    user_prompt: str,
    search_query: str,
    context_blocks: str,
) -> tuple[str, dict]:
    model, tokenizer = _load_model()
    hypothesis_id = _hypothesis_id()
    base_prompt = build_llama_hypothesis_prompt(
        user_prompt=user_prompt,
        search_query=search_query,
        context_blocks=context_blocks,
        hypothesis_id=hypothesis_id,
    )
    total_attempts = 1 + NOVELTY_REWRITE_ATTEMPTS
    novelty_attempts: list[dict[str, Any]] = []

    prompt = base_prompt
    text = ""
    last_prompt_tokens = 0
    for attempt_idx in range(total_attempts):
        text, prompt_tokens = _generate_text(model, tokenizer, prompt)
        last_prompt_tokens = prompt_tokens
        contrived = is_contrived_hypothesis(text)
        novelty_attempts.append(
            {
                "attempt": attempt_idx + 1,
                "contrived": contrived,
                "chars": len(text),
                "prompt_tokens": prompt_tokens,
            }
        )
        if not contrived:
            break
        if attempt_idx == total_attempts - 1:
            raise HypothesisNoveltyError(
                "Novelty validation failed: generated hypothesis still appears to stitch existing theories. "
                "Rephrase the prompt toward one original mechanism and rerun."
            )
        prompt = _novelty_retry_prompt(base_prompt, text)

    meta = {
        "model_id": MODEL_ID,
        "tokenizer_id": TOKENIZER_ID,
        "hypothesis_id": hypothesis_id,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "prompt_tokens": last_prompt_tokens,
        "novelty_validation": "passed",
        "novelty_attempts": novelty_attempts,
        "novelty_rewrite_attempts_used": max(0, len(novelty_attempts) - 1),
    }
    return text, meta
