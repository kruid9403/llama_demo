import os, threading
from dotenv import load_dotenv
from transformers import TextIteratorStreamer
from messages.system_prompt import get_system_prompt


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

os.environ.setdefault("HF_TOKEN", "sk-prod-onwALVUHNAKLSJRFNWJKGILUVHDAFKLJVNFGIALDRhvn")
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.7,max_split_size_mb:128",
)

load_dotenv()

max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "512"))
temperature = float(os.getenv("TEMPERATURE", "0.7"))

MODEL_ID = os.getenv("MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
TOKENIZER_ID = os.getenv("TOKENIZER_ID", MODEL_ID)
USE_8BIT = os.getenv("USE_8BIT", "1") == "1"

device = "cuda" if os.getenv("USE_CUDA", "1") == "1" else "cpu"

# tokenizer = AutoTokenizer.from_pretrained(
#     TOKENIZER_ID,
#     token=os.getenv("HF_TOKEN"),
# )

# tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token_id = tokenizer.eos_token_id

# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_ID,
#     load_in_8bit=USE_8BIT,
#     device_map="auto",
#     token=os.getenv("HF_TOKEN"),
#     offload_buffers=True,
#     dtype="auto"
# )

def load_model():
    print(f"Device: {device} | USE_8BIT={USE_8BIT}")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        token=os.getenv("HF_TOKEN"),
    )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model_kwargs = dict(
        device_map="auto",
        token=os.getenv("HF_TOKEN"),
        offload_buffers=True,
        low_cpu_mem_usage=True,
    )
    if USE_8BIT:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        model_kwargs["torch_dtype"] = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    # Avoid invalid-flag warnings when callers run deterministic decoding.
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    return model, tokenizer

def _compute_eot_ids(tokenizer) -> list[int]:
    eot_ids = [tokenizer.eos_token_id]
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    except Exception:
        eot_id = None
    if eot_id is not None and eot_id != tokenizer.eos_token_id:
        eot_ids.append(eot_id)
    return eot_ids


if __name__ == "__main__":
    model, tokenizer = load_model()
    print("Model and tokenizer loaded successfully.")
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    system_prompt_text = get_system_prompt()

    messages = [
        {"role": "system", "content": system_prompt_text},
    ]

    while True:
        text = input(">>> ").strip()
        if text.lower() == "exit":
            break
        print(f"You entered: {text}")
        enc = tokenizer(text, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        _eot_ids = _compute_eot_ids(tokenizer)

        generate_kwargs = dict(
            inputs=input_ids,
            attention_mask=attention_mask,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            eos_token_id=_eot_ids or [tokenizer.eos_token_id],
            pad_token_id=tokenizer.pad_token_id,
        )

        thread = threading.Thread(target=model.generate, kwargs=generate_kwargs)
        thread.start()

        for text in streamer:
            print(text, end="", flush=True)

        messages.append({"role": "user", "content": text})
        messages.append({"role": "assistant", "content": streamer})
