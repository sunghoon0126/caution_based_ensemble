"""Model-specific input serialization for scalar reward models."""

SKYWORK_MODELS = (
    "Skywork/Skywork-Reward-V2-Qwen3-1.7B",
    "Skywork/Skywork-Reward-V2-Llama-3.2-3B",
)


def resolve_backend(model_name, backend="auto"):
    if backend != "auto":
        return backend
    return "skywork" if model_name in SKYWORK_MODELS else "pair"


def tokenize_rewards(tokenizer, records, backend, max_length=None):
    for record in records:
        if not all(isinstance(record.get(k), str) for k in ("prompt", "response")):
            raise ValueError("Reward inputs require string prompt and response")
    if backend == "pair":
        kwargs = dict(padding=True, truncation=True, return_tensors="pt")
        if max_length is not None:
            kwargs["max_length"] = max_length
        return tokenizer([r["prompt"] for r in records], [r["response"] for r in records], **kwargs)
    if backend != "skywork":
        raise ValueError(f"Unsupported reward backend: {backend}")
    formatted = []
    for record in records:
        # Official Skywork example: user/assistant only; strip a template BOS
        # before tokenizer adds its own special tokens (needed for Llama).
        text = tokenizer.apply_chat_template([
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]},
        ], tokenize=False, add_generation_prompt=False)
        if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
            text = text[len(tokenizer.bos_token):]
        formatted.append(text)
    # Reject overlong conversations rather than silently dropping the answer.
    encoded = tokenizer(formatted, truncation=False, padding=False)
    limit = max_length or 16384
    if any(len(ids) > limit for ids in encoded["input_ids"]):
        raise ValueError(f"Skywork input exceeds {limit} tokens; inspect candidate or increase --max-length")
    return tokenizer.pad(encoded, padding=True, return_tensors="pt")
