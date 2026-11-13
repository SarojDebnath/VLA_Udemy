"""Shared closed-loop evaluation harness for TableTop policies."""
import numpy as np
from scipy.stats import norm


def wilson(k, n, conf=0.95):
    if n == 0:
        return 0.0, 1.0
    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z / den * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float(c - h), float(c + h)


def evaluate(env, policy, n_episodes=50, seed=10_000, max_steps=None, record=0, camera="front", on_step=None, on_reset=None):
    """policy: object with reset() and act(obs) -> 7-D action. Returns dict with success rate, CI, per-task rates, videos."""
    results, videos = [], []
    for ep in range(n_episodes):
        obs = env.reset(seed=seed + ep)
        if on_reset is not None:
            on_reset(env)
            obs = env.observe()
        if hasattr(policy, "reset"):
            policy.reset()
        frames, done, steps = [], False, 0
        while not done:
            a = policy.act(obs)
            if on_step is not None:
                a = on_step(env, obs, a)
            obs, _, done, info = env.step(a)
            steps += 1
            if ep < record:
                frames.append(obs["images"][camera] if camera in obs["images"] else env.render(camera))
            if max_steps and steps >= max_steps:
                break
        results.append(dict(task=env.task["type"], instruction=env.task["instruction"], success=bool(info["success"]), steps=steps))
        if ep < record:
            videos.append(frames)
    k = sum(r["success"] for r in results)
    per_task = {}
    for r in results:
        per_task.setdefault(r["task"], []).append(r["success"])
    return dict(success=k / n_episodes, ci=wilson(k, n_episodes), n=n_episodes,
                per_task={t: float(np.mean(v)) for t, v in per_task.items()}, episodes=results, videos=videos)


class ExpertPolicy:
    """Wraps the scripted expert in the policy interface (re-created at every reset)."""

    def __init__(self, env, **kw):
        from .expert import ScriptedExpert
        self.env, self.kw, self.cls = env, kw, ScriptedExpert
        self.expert = None

    def reset(self):
        self.expert = self.cls(self.env, **self.kw)

    def act(self, obs):
        return self.expert.act(obs)


def need(path, made_by):
    """Stop with a helpful message when a notebook needs weights trained in ANOTHER notebook."""
    import os
    if not os.path.exists(path):
        section = made_by.split()[0]
        raise FileNotFoundError(f"\n\nThis notebook uses '{path}', trained in {made_by}.\n"
                                f"Either run that notebook first, or download the course weights:\n"
                                f"    python tools/weights.py get {section}\n")
    return path
