"""Actual stateless SGD; each call starts at the immutable base checkpoint."""

import math

from ensemble.models.scalar_reward_inputs import tokenize_rewards
from ensemble.evaluation.baseline6.common import finite


def functional_adapt(model, inputs, target, learning_rate, num_steps=1, parameter_names=None):
    import torch
    from torch.func import functional_call

    if not math.isfinite(learning_rate) or learning_rate <= 0 or num_steps < 0:
        raise ValueError("Positive finite learning rate and nonnegative steps required")
    model.eval()
    base = dict(model.named_parameters())
    names = list(base) if parameter_names is None else list(parameter_names)
    if not names or any(name not in base for name in names):
        raise ValueError("Empty or unknown update parameter scope")
    # Detached views are never mutated. Each SGD result owns fresh storage.
    params = {name: tensor.detach().requires_grad_(name in names) for name, tensor in base.items()}
    buffers = {name: tensor.detach().clone() for name, tensor in model.named_buffers()}

    def forward():
        output = functional_call(model, (params, buffers), (), inputs, strict=True)
        logits = output.logits if hasattr(output, "logits") else output
        if logits.numel() != 1:
            raise ValueError("Expected exactly one scalar candidate reward")
        return logits.reshape(()).float()

    with torch.enable_grad():
        reward = forward()
        before = finite(reward.detach().item())
        target = finite(target)
        loss_before = finite(0.5 * (before - target) ** 2)
        norms = []
        for _ in range(num_steps):
            loss = 0.5 * (reward - target).square()
            finite(loss.detach().item())
            gradients = torch.autograd.grad(loss, [params[name] for name in names], allow_unused=True)
            norm2 = 0.0
            updated = dict(params)
            with torch.no_grad():
                for name, gradient in zip(names, gradients):
                    if gradient is None:
                        continue
                    # Accumulate in float64 for diagnostic overflow detection.
                    norm2 += finite(gradient.double().square().sum().item())
                    updated[name] = (params[name] - learning_rate * gradient).detach().requires_grad_(True)
                    if not torch.isfinite(updated[name]).all():
                        raise FloatingPointError(f"Non-finite updated parameter: {name}")
            norms.append(finite(math.sqrt(norm2)))
            params = updated
            del gradients, loss
            reward = forward()
        after = finite(reward.detach().item())
    return dict(reward_before=before, reward_after=after, reward_change=finite(after - before),
                gradient_norm=norms[0] if norms else 0.0, gradient_norms=norms,
                dummy_loss_before=loss_before, dummy_loss_after=finite(0.5 * (after - target) ** 2))


class RewardModelAdapter:
    def __init__(self, model_name, device="cuda:0", dtype="float32", backend="auto", revision="main", max_length=None):
        from ensemble.evaluation.score_reference_candidates import load_reward_model
        from ensemble.models.scalar_reward_inputs import resolve_backend
        self.settings = dict(backend=resolve_backend(model_name, backend), dtype=dtype,
                             revision=revision, max_length=max_length)
        self.tokenizer, self.model = load_reward_model(model_name, device, self.settings)
        self.device = device
        self.model_name = model_name

    def inputs(self, prompt, response):
        tokens = tokenize_rewards(self.tokenizer, [dict(prompt=prompt, response=response)],
                                  self.settings["backend"], self.settings["max_length"])
        return {name: value.to(self.device) for name, value in tokens.items()}

    def score(self, prompt, response):
        import torch
        with torch.no_grad():
            return finite(self.model(**self.inputs(prompt, response)).logits.item())

    def adapted_score(self, prompt, response, target, learning_rate, num_steps=1, update_scope="full"):
        names = None
        if update_scope == "head_only":
            # Explicit architecture mapping; do not guess silently for unknown RMs.
            if self.model.config.model_type == "deberta-v2":
                names = [n for n, _ in self.model.named_parameters() if n.startswith(("pooler.", "classifier."))]
            elif self.model.config.model_type in ("qwen3", "llama"):
                names = [n for n, _ in self.model.named_parameters() if n.startswith("score.")]
            else:
                raise ValueError("head_only mapping unavailable for this architecture")
        elif update_scope != "full":
            raise ValueError("Supported scopes: full, head_only")
        return functional_adapt(self.model, self.inputs(prompt, response), target, learning_rate, num_steps, names)
