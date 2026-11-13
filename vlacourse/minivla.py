"""MiniVLA: the OpenVLA recipe in miniature (S10 project 103).

SmolVLM-256M (SigLIP-style vision encoder -> projector -> SmolLM2 decoder) is fine-tuned with LoRA to emit
7 discrete action tokens per step. The last 256 vocabulary ids are repurposed as action bins (OpenVLA overwrites
the 256 least-used Llama tokens). Bins are fit per dimension on the 1st..99th percentile of the training actions.
"""
import numpy as np
import torch
from PIL import Image

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


class ActionBins:
    def __init__(self, actions, n_bins=256, vocab_size=49280):
        self.lo = np.percentile(actions, 1, 0)
        self.hi = np.percentile(actions, 99, 0)
        self.hi = np.where(self.hi - self.lo < 1e-6, self.lo + 1e-6, self.hi)
        self.n, self.first_id = n_bins, vocab_size - n_bins

    def encode(self, a):
        z = (np.asarray(a, np.float64) - self.lo) / (self.hi - self.lo)
        # clip the BIN INDEX, not z: with float32 inputs numpy 2 keeps float32, where 1 - 1e-9 == 1.0 -> bin 256 (out of vocab)
        return np.clip((z * self.n).astype(np.int64), 0, self.n - 1) + self.first_id

    def decode(self, ids):
        return (np.asarray(ids) - self.first_id + 0.5) / self.n * (self.hi - self.lo) + self.lo


class MiniVLA:
    def __init__(self, actions, device="cuda", lora_r=16, train_vision=True, dtype=torch.bfloat16, image_size=256, grad_ckpt=True,
                 n_actions=1):
        """n_actions: how many future 7-D actions one prediction holds (a token action CHUNK of 7*n_actions tokens)."""
        self.n_actions = n_actions
        actions = np.asarray(actions).reshape(-1, 7)
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from peft import LoraConfig, get_peft_model
        self.device = device
        self.proc = AutoProcessor.from_pretrained(MODEL_ID)
        self.proc.tokenizer.padding_side = "left"
        self.size = {"longest_edge": image_size}
        self.proc.image_seq_len = int(64 * (image_size / 512) ** 2)
        base = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=dtype)
        if grad_ckpt:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            base.enable_input_require_grads()
        # The LAST 256 ids of the full vocab (49024..49279) include SmolVLM's added special tokens (<image> = 49190, ...):
        # an action landing there would be read as an extra image placeholder. Use the last 256 REGULAR BPE tokens
        # (tokenizer.vocab_size excludes added tokens), like OpenVLA's overwrite of rarely-used vocabulary.
        self.bins = ActionBins(actions, vocab_size=self.proc.tokenizer.vocab_size)
        targets = r".*(text_model|vision_model).*\.(q_proj|k_proj|v_proj|o_proj|out_proj)$" if train_vision else r".*text_model.*\.(q_proj|k_proj|v_proj|o_proj)$"
        cfg = LoraConfig(r=lora_r, lora_alpha=lora_r, lora_dropout=0.05, target_modules=targets, modules_to_save=["lm_head"])
        self.model = get_peft_model(base, cfg).to(device)
        self.image_token_id = base.config.image_token_id
        self.dtype, self._text_cache = dtype, {}

    def prompt(self, instruction):
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"What action should the robot take to {instruction}?"}]}]
        return self.proc.apply_chat_template(msgs, add_generation_prompt=True)

    def _text(self, instruction):
        if instruction not in self._text_cache:
            ids = self.proc(text=[self.prompt(instruction)], images=[[Image.new("RGB", (8, 8))]], return_tensors="pt",
                            do_image_splitting=False, size=self.size, max_image_size=self.size)["input_ids"][0]
            self._text_cache[instruction] = ids
        return self._text_cache[instruction]

    def pixels(self, images):
        """uint8 HxWx3 numpy images -> normalized (B, 1, 3, S, S) on the GPU (fast path, no PIL)."""
        x = torch.from_numpy(np.stack(images)).to(self.device).permute(0, 3, 1, 2).float().div(255)
        x = torch.nn.functional.interpolate(x, size=(self.size["longest_edge"],) * 2, mode="bilinear", align_corners=False)
        return ((x - 0.5) / 0.5).to(self.dtype).unsqueeze(1)

    def batch(self, images, instructions, actions=None):
        seqs = [self._text(t) for t in instructions]
        L = max(len(s) for s in seqs)
        pad = self.proc.tokenizer.pad_token_id
        ids = torch.stack([torch.cat([torch.full((L - len(s),), pad, dtype=s.dtype), s]) for s in seqs])
        attn = torch.stack([torch.cat([torch.zeros(L - len(s), dtype=torch.long), torch.ones(len(s), dtype=torch.long)]) for s in seqs])
        x = {"input_ids": ids, "attention_mask": attn}
        if actions is not None:
            act = torch.from_numpy(self.bins.encode(np.asarray(actions).reshape(len(seqs), -1, 7)).reshape(len(seqs), -1))
            x["labels"] = torch.cat([torch.full_like(ids, -100), act], 1)
            x["input_ids"] = torch.cat([ids, act], 1)
            x["attention_mask"] = torch.cat([attn, torch.ones_like(act)], 1)
        x = {k: v.to(self.device) for k, v in x.items()}
        x["pixel_values"] = self.pixels(images)
        return x

    def loss(self, images, instructions, actions):
        b = self.batch(images, instructions, actions)
        labels = b.pop("labels")
        n = 7 * self.n_actions
        logits = self.model(**b).logits[:, -n - 1:-1]
        target = labels[:, -n:]
        loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1))
        acc = (logits.argmax(-1) == target).float().mean()
        return loss, acc

    @torch.no_grad()
    def predict(self, images, instructions, decode="mode"):
        """Greedy decoding of 7 action tokens, restricted to the action-bin vocabulary.
        decode="mode": each dimension is the most likely bin (OpenVLA). decode="mean": the probability-weighted average
        bin (the token head's analogue of MSE regression); the greedy token is still what conditions the next dimension."""
        b = self.batch(images, instructions)
        out = self.model(**b, use_cache=True)
        past, logits, ids, means = out.past_key_values, out.logits[:, -1], [], []
        lo, n = self.bins.first_id, self.bins.n
        attn = b["attention_mask"]
        centers = torch.arange(n, device=logits.device, dtype=torch.float32)
        n_tok = 7 * self.n_actions
        for k in range(n_tok):
            p = logits.float()[:, lo:lo + n].softmax(-1)
            nxt = p.argmax(-1, keepdim=True) + lo
            ids.append(nxt); means.append((p * centers).sum(-1, keepdim=True) + lo)
            if k == n_tok - 1:
                break
            attn = torch.cat([attn, torch.ones_like(nxt)], 1)
            out = self.model(input_ids=nxt, attention_mask=attn, past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1]
        tok = torch.cat(means if decode == "mean" else ids, 1).cpu().numpy().reshape(-1, self.n_actions, 7)
        out = self.bins.decode(tok)
        return out[:, 0] if self.n_actions == 1 else out          # (B, 7) or a chunk (B, n_actions, 7)


class MiniVLAPolicy:
    """ds: optional TableTopDataset (action_mode 'rel', horizon 1) when the bins encode commanded targets RELATIVE to the
    gripper instead of raw per-step deltas; the prediction is converted back with ds.to_absolute / ds.to_command."""

    def __init__(self, vla, env, camera="front", instruction_fn=None, ds=None, decode="mode", execute=None):
        """execute: with a chunk model (n_actions > 1), how many actions of each chunk to run before predicting again."""
        self.vla, self.env, self.camera, self.ds, self.decode = vla, env, camera, ds, decode
        self.instruction_fn = instruction_fn or (lambda task: task["instruction"])
        self.execute, self.queue = execute or vla.n_actions, []

    def reset(self):
        self.queue = []

    def act(self, obs):
        if not self.queue:
            p = self.vla.predict([obs["images"][self.camera]], [self.instruction_fn(self.env.task)], decode=self.decode)[0]
            chunk = p.reshape(-1, 7).copy()
            if self.ds is not None:
                chunk = self.ds.to_absolute(chunk, obs)
            self.queue = list(chunk[: self.execute])
        a = self.queue.pop(0)
        a = self.ds.to_command(a, self.env) if self.ds is not None else np.array(a, np.float32)
        a[6] = float(a[6] > 0.5)
        return a


class FlowActionExpert(torch.nn.Module):
    """π0-style action expert: noisy action-chunk tokens (+ flow time) attend to the VLM prefix and the robot state."""

    def __init__(self, prefix_dim, act_dim=7, horizon=16, d=256, layers=4, heads=8, state_dim=8):
        super().__init__()
        nn = torch.nn
        self.H, self.A = horizon, act_dim
        self.prefix_proj, self.state_proj = nn.Linear(prefix_dim, d), nn.Linear(state_dim, d)
        self.act_in, self.time_mlp = nn.Linear(act_dim, d), nn.Sequential(nn.Linear(128, d), nn.SiLU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.randn(1, horizon, d) * 0.02)
        self.dec = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, heads, 4 * d, 0.0, batch_first=True, norm_first=True), layers)
        self.out = nn.Linear(d, act_dim)

    def forward(self, a_tau, tau, prefix, prefix_mask, state):
        from .chunk_policies import sinusoidal
        x = self.act_in(a_tau) + self.pos + self.time_mlp(sinusoidal(tau * 100, 128))[:, None]
        mem = torch.cat([self.state_proj(state)[:, None], self.prefix_proj(prefix)], 1)
        pad = torch.cat([torch.zeros_like(prefix_mask[:, :1], dtype=torch.bool), prefix_mask == 0], 1)
        return self.out(self.dec(x, mem, memory_key_padding_mask=pad))


class MiniVLAFlow(MiniVLA):
    """MiniVLA v2: the same SmolVLM backbone, but continuous action chunks from a flow-matching expert (S11)."""

    def __init__(self, actions, states, horizon=16, device="cuda", **kw):
        super().__init__(actions, device=device, **kw)
        self.H = horizon
        self.a_mean, self.a_std = actions.mean(0), actions.std(0) + 1e-6
        self.s_mean, self.s_std = states.mean(0), states.std(0) + 1e-6
        dim = self.model.base_model.model.config.text_config.hidden_size
        self.expert = FlowActionExpert(dim, horizon=horizon).to(device)

    def parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad] + list(self.expert.parameters())

    def prefix(self, images, instructions):
        b = self.batch(images, instructions)
        out = self.model(**b, output_hidden_states=True)
        return out.hidden_states[-1].float(), b["attention_mask"]

    def _norm(self, x, m, s):
        return torch.as_tensor((x - m) / s, dtype=torch.float32, device=self.device)

    def loss(self, images, instructions, chunks, states):
        prefix, mask = self.prefix(images, instructions)
        a1 = self._norm(chunks, self.a_mean, self.a_std)
        a0 = torch.randn_like(a1)
        tau = torch.rand(len(a1), device=self.device)
        a_tau = (1 - tau)[:, None, None] * a0 + tau[:, None, None] * a1
        v = self.expert(a_tau, tau, prefix, mask, self._norm(states, self.s_mean, self.s_std))
        return torch.nn.functional.mse_loss(v, a1 - a0)

    @torch.no_grad()
    def predict_chunk(self, images, instructions, states, steps=10):
        prefix, mask = self.prefix(images, instructions)
        s = self._norm(states, self.s_mean, self.s_std)
        x = torch.randn(len(images), self.H, 7, device=self.device)
        for k in range(steps):
            tau = torch.full((len(images),), k / steps, device=self.device)
            x = x + self.expert(x, tau, prefix, mask, s) / steps
        return x.cpu().numpy() * self.a_std + self.a_mean


class MiniVLAFlowPolicy:
    """ds: optional TableTopDataset whose action_mode defines what the chunk means ('rel' = commanded targets relative
    to the gripper pose; converted back to env commands with ds.to_absolute / ds.to_command). None = raw deltas."""

    def __init__(self, vla, env, execute=8, steps=10, camera="front", ds=None):
        self.vla, self.env, self.k, self.steps, self.camera, self.ds = vla, env, execute, steps, camera, ds
        self.queue = []

    def reset(self):
        self.queue = []

    def act(self, obs):
        if not self.queue:
            chunk = self.vla.predict_chunk([obs["images"][self.camera]], [self.env.task["instruction"]], obs["state"][None], self.steps)[0]
            if self.ds is not None:
                chunk = self.ds.to_absolute(chunk, obs)
            self.queue = list(chunk[: self.k])
        a = self.queue.pop(0)
        a = self.ds.to_command(a, self.env) if self.ds is not None else a.copy()
        a[6] = float(a[6] > 0.5)
        return a


def train_flow(ds, n_eps=None, rank=16, steps=2000, bs=16, lr=2e-4, seed=0, device="cuda", log_every=0):
    """Fine-tune MiniVLAFlow (SmolVLM + LoRA + flow action expert) on the first n_eps episodes of a TableTopDataset
    (action_mode 'rel', horizon H). Normalization stats come from the whole dataset. Returns the model in eval mode."""
    import time
    n = len(ds.eps) if n_eps is None else n_eps
    chunks = [ds.episode_chunks(i) for i in range(len(ds.eps))]
    vla = MiniVLAFlow(np.concatenate([c.reshape(-1, 7) for c in chunks]), np.concatenate([e["state"] for e in ds.eps]),
                      horizon=ds.horizon, device=device, lora_r=rank)
    frames = [(ds.eps[i]["front"][t], ds.eps[i]["instruction"], chunks[i][t], ds.eps[i]["state"][t])
              for i in range(n) for t in range(len(ds.eps[i]["action"]))]
    params = vla.parameters()
    opt = torch.optim.AdamW(params, lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    rng, t0 = np.random.default_rng(seed), time.time()
    torch.manual_seed(seed)
    vla.model.train()
    for step in range(1, steps + 1):
        b = [frames[j] for j in rng.integers(0, len(frames), bs)]
        loss = vla.loss([x[0] for x in b], [x[1] for x in b], np.stack([x[2] for x in b]), np.stack([x[3] for x in b]))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        if log_every and step % log_every == 0:
            print(f"  step {step}/{steps} flow loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
    vla.model.eval()
    return vla
