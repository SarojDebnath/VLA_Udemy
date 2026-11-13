"""Share-ready videos (LinkedIn / slides / course promo): captions, plan panels, grids, title cards, H.264 export.

LinkedIn-friendly defaults: 1080x1080 (square) or 1280x720 (16:9), H.264 + yuv420p, 30 fps, < 10 min.
"""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

BG = (18, 20, 28)            # dark background
FG = (236, 239, 244)         # main text
MUTED = (140, 148, 165)      # secondary text
ACCENT = (64, 196, 255)      # highlight / progress
GOOD = (80, 220, 140)        # done / success

_FONT_CACHE = {}


def font(size, bold=False):
    """A TrueType font that exists on Windows, Linux/Colab and macOS (falls back to PIL's scalable default)."""
    key = (size, bold)
    if key not in _FONT_CACHE:
        names = []
        try:   # matplotlib ships DejaVu Sans everywhere, and it has the glyphs we use (✓ ▶ →)
            from matplotlib import font_manager as fm
            names.append(fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")))
        except Exception:
            pass
        names += (["DejaVuSans-Bold.ttf", "arialbd.ttf"] if bold else []) + ["DejaVuSans.ttf", "arial.ttf", "Helvetica.ttc"]
        f = None
        for n in names:
            try:
                f = ImageFont.truetype(n, size); break
            except OSError:
                continue
        _FONT_CACHE[key] = f or ImageFont.load_default(size=size)
    return _FONT_CACHE[key]


def fitted(draw, text, max_w, size, bold=False, min_size=10):
    """Largest font <= size whose rendering of text fits in max_w pixels."""
    while size > min_size and draw.textlength(text, font=font(size, bold)) > max_w:
        size -= 1
    return font(size, bold)


def _canvas(img):
    im = Image.fromarray(np.asarray(img).astype(np.uint8)).convert("RGBA")
    return im, Image.new("RGBA", im.size, (0, 0, 0, 0))


def _done(im, layer):
    return np.asarray(Image.alpha_composite(im, layer).convert("RGB"))


def caption(img, text, where="bottom", size=None, pad=None, alpha=170, color=FG):
    """Semi-transparent bar with text across the top or bottom of the frame."""
    im, layer = _canvas(img)
    W, H = im.size
    size = size or max(14, H // 24); pad = pad or size // 2
    d = ImageDraw.Draw(layer)
    f = fitted(d, text, W - 3 * pad, size, bold=True)
    h = size + 2 * pad
    y0 = 0 if where == "top" else H - h
    d.rectangle([0, y0, W, y0 + h], fill=(0, 0, 0, alpha))
    d.text((pad * 1.5, y0 + pad), text, font=f, fill=color + (255,))
    return _done(im, layer)


def badge(img, text, corner="tr", size=None, color=ACCENT):
    """Small rounded label in a corner (e.g. 'step 3/10', '2x speed', 'Panda')."""
    im, layer = _canvas(img)
    W, H = im.size
    size = size or max(12, H // 30)
    d = ImageDraw.Draw(layer); f = font(size, bold=True)
    tw = d.textlength(text, font=f); p = size // 2
    w, h = tw + 2 * p, size + 2 * p
    x = W - w - p if corner[1] == "r" else p
    y = p if corner[0] == "t" else H - h - p
    d.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=color + (230,))
    d.text((x + p, y + p * 0.8), text, font=f, fill=(10, 12, 18, 255))
    return _done(im, layer)


def progress(img, frac, height=6, color=ACCENT):
    out = np.asarray(img).copy()
    out[-height:, : int(out.shape[1] * float(np.clip(frac, 0, 1)))] = color
    return out


def title_card(title, subtitle="", size=(720, 1280), n=45, footer=""):
    """n identical frames of a dark title slide."""
    H, W = size
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    ft, fs, ff = font(max(24, H // 12), bold=True), font(max(16, H // 26)), font(max(12, H // 40))
    tw = d.textlength(title, font=ft)
    d.text(((W - tw) / 2, H * 0.36), title, font=ft, fill=FG)
    d.rectangle([(W - 120) / 2, H * 0.36 + H // 12 + 24, (W + 120) / 2, H * 0.36 + H // 12 + 28], fill=ACCENT)
    if subtitle:
        for i, line in enumerate(subtitle.split("\n")):
            sw = d.textlength(line, font=fs)
            d.text(((W - sw) / 2, H * 0.36 + H // 12 + 48 + i * (H // 22)), line, font=fs, fill=MUTED)
    if footer:
        fw = d.textlength(footer, font=ff)
        d.text(((W - fw) / 2, H - H // 12), footer, font=ff, fill=MUTED)
    return [np.asarray(im)] * n


def side_panel(lines, active, size=(720, 320), header="PLAN", done_color=GOOD):
    """A list panel (e.g. a symbolic plan) with finished items ticked and the active one highlighted."""
    H, W = size
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    fh, fl = font(max(14, H // 28), bold=True), font(max(12, H // 36))
    d.text((24, 24), header, font=fitted(d, header, W - 48, fh.size, bold=True), fill=ACCENT)
    row = (H - 90) / max(len(lines), 1)
    row = min(row, H // 14)
    for i, line in enumerate(lines):
        y = 80 + i * row
        if i == active:
            d.rounded_rectangle([12, y - 6, W - 12, y + row - 8], radius=8, fill=(40, 60, 90))
            col, mark = FG, "▶"           # ▶
        elif i < active:
            col, mark = done_color, "✓"   # ✓
        else:
            col, mark = MUTED, " "
        d.text((24, y), mark, font=fl, fill=col)
        d.text((52, y), line, font=fitted(d, line, W - 76, fl.size), fill=col)
    return np.asarray(im)


def grid(clips, cols, labels=None, gap=6, hold=0):
    """Tile several clips (lists of equal-size frames) into one video. Shorter clips freeze on their last frame
    (with a green 'done' badge), so every clip is shown finishing. Optional per-clip caption."""
    T = max(len(c) for c in clips) + hold
    rows = int(np.ceil(len(clips) / cols))
    h, w = np.asarray(clips[0][0]).shape[:2]
    out = []
    for t in range(T):
        canvas = np.full((rows * h + (rows + 1) * gap, cols * w + (cols + 1) * gap, 3), BG, np.uint8)
        for k, c in enumerate(clips):
            f = c[min(t, len(c) - 1)]
            if labels:
                f = caption(f, labels[k], size=max(11, h // 20))
            if t >= len(c):
                f = badge(f, "✓ done", color=GOOD, size=max(11, h // 22))
            r, cc = divmod(k, cols)
            y, x = gap + r * (h + gap), gap + cc * (w + gap)
            canvas[y:y + h, x:x + w] = f
        out.append(canvas)
    return out


def fit(img, size):
    """Letterbox an image into (H, W) on the dark background."""
    H, W = size
    im = Image.fromarray(np.asarray(img))
    s = min(W / im.width, H / im.height)
    im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    out = Image.new("RGB", (W, H), BG); out.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
    return np.asarray(out)


def export(frames, path, fps=30, gif=False, gif_width=480, gif_fps=10):
    """H.264 + yuv420p MP4 (plays on LinkedIn, phones, PowerPoint) and an optional small looping GIF."""
    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frames = [np.asarray(f)[: np.asarray(f).shape[0] // 2 * 2, : np.asarray(f).shape[1] // 2 * 2] for f in frames]
    imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=2,
                    pixelformat="yuv420p", ffmpeg_params=["-movflags", "+faststart"])
    out = [path]
    if gif:
        step = max(1, round(fps / gif_fps))
        small = []
        for f in frames[::step]:
            im = Image.fromarray(f); s = gif_width / im.width
            small.append(np.asarray(im.resize((gif_width, int(im.height * s) // 2 * 2), Image.LANCZOS)))
        g = os.path.splitext(path)[0] + ".gif"
        imageio.mimsave(g, small, duration=1000 / gif_fps, loop=0)
        out.append(g)
    for p in out:
        print(f"saved {p}  ({os.path.getsize(p) / 1e6:.1f} MB)")
    return out


# ---------------------------------------------------------------- policy videos (S06+)
BAD = (240, 90, 90)


def rollout(env, policy, seed, camera="showcase", size=(360, 480), fps=30, max_steps=None, instruction=None, on_reset=None,
            on_step=None, reset_kw=None, overlay=None):
    """One episode recorded in HD. Returns (frames, success, instruction).
    on_reset(env) / on_step(env, obs, a) -> a work like in vlacourse.evaluate; reset_kw goes to env.reset (e.g. task=...)."""
    obs = env.reset(seed=seed, **(reset_kw or {}))
    if on_reset is not None:
        on_reset(env); obs = env.observe()
    if hasattr(policy, "reset"):
        policy.reset()
    env.start_video(camera, size, fps, overlay=overlay)
    done, steps = False, 0
    while not done:
        a = policy.act(obs)
        if on_step is not None:
            a = on_step(env, obs, a)
        obs, _, done, info = env.step(a); steps += 1
        if max_steps and steps >= max_steps:
            break
    frames = env.stop_video()
    return frames, bool(info["success"]), instruction or env.task["instruction"]


def _finish(frames, ok, hold=15):
    """Freeze the last frame with a success/failure badge."""
    end = badge(frames[-1], "✓ success" if ok else "✗ failed", color=GOOD if ok else BAD)
    return list(frames) + [end] * hold


def watch(env, policy, seeds=(0, 1, 2, 3), title="policy rollouts", cols=2, size=(360, 480), fps=30,
          export=True, caption=True, **kw):
    """Run the policy on a few seeds and show a captioned grid video (instruction on each tile, success/fail badge).
    The video is embedded in the notebook (downloadable) and saved to outputs/<title>.mp4."""
    from .viz import show_video, _slug
    clips, labels, oks = [], [], []
    for sd in seeds:
        fr, ok, ins = rollout(env, policy, sd, size=size, fps=fps, **kw)
        clips.append(_finish(fr, ok)); labels.append(ins if caption else ""); oks.append(ok)
    frames = grid(clips, cols=cols, labels=labels if caption else None)
    frames = [progress(f, (i + 1) / len(frames)) for i, f in enumerate(frames)]
    print(f"{title}: {sum(oks)}/{len(oks)} successful")
    show_video(frames, fps=fps, title=title)
    if export:
        os.makedirs("outputs", exist_ok=True)
        _quiet_export(frames, f"outputs/{_slug(title)}.mp4", fps)
    return oks


def compare(env, policies, seed=0, title="comparison", size=(360, 480), fps=30, export=True, per_policy_kw=None, footer=None, **kw):
    """The SAME episode (same seed, same layout, same instruction) run by several policies, side by side."""
    from .viz import show_video, _slug
    clips, oks = [], []
    for name, pol in policies.items():
        extra = (per_policy_kw or {}).get(name, {})
        fr, ok, ins = rollout(env, pol, seed, size=size, fps=fps, **{**kw, **extra})
        fr = [caption(f, name, where="top") for f in fr]
        clips.append(_finish(fr, ok)); oks.append(ok)
    T = max(len(c) for c in clips)
    clips = [c + [c[-1]] * (T - len(c)) for c in clips]
    frames = [np.hstack([c[t] for c in clips]) for t in range(T)]
    text = f"“{ins}”" if footer is None else footer
    if text:
        frames = [caption(f, text, size=max(14, size[0] // 22)) for f in frames]
    print(title + ": " + " | ".join(f"{n}: {'success' if o else 'failed'}" for n, o in zip(policies, oks)))
    show_video(frames, fps=fps, title=title)
    if export:
        os.makedirs("outputs", exist_ok=True)
        _quiet_export(frames, f"outputs/{_slug(title)}.mp4", fps)
    return oks


def pick_seed(res_a, res_b, base_seed, default=0):
    """First episode index where policy A succeeded and B failed (two vlacourse.evaluate results on the same seeds),
    so a side-by-side video shows the difference instead of a random tie. Returns base_seed + index."""
    for i, (ea, eb) in enumerate(zip(res_a["episodes"], res_b["episodes"])):
        if ea["success"] and not eb["success"]:
            return base_seed + i
    return base_seed + default


def _quiet_export(frames, path, fps):
    import imageio
    frames = [np.asarray(f)[: np.asarray(f).shape[0] // 2 * 2, : np.asarray(f).shape[1] // 2 * 2] for f in frames]
    imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=8, macro_block_size=2,
                    pixelformat="yuv420p", ffmpeg_params=["-movflags", "+faststart"])


def force_gauge(limit=25.0):
    """overlay for rollout/compare: live wrist force, green under `limit`, red above (needs TableTop(wrist_ft=True))."""
    def f(env):
        n = float(np.linalg.norm(env.wrist_force()))
        return f"wrist force {n:4.0f} N", (GOOD if n < limit else BAD)
    return f


def side_by_side(clips, labels, title="side by side", fps=30, footer=None, scale=1, export=True):
    """Any list of frame clips (not just TableTop rollouts) shown side by side with a label on each, padded to equal length.
    scale: integer nearest-neighbour upscaling for tiny frames (e.g. 64x64 world-model images)."""
    from .viz import show_video, _slug
    if scale > 1:
        clips = [[np.repeat(np.repeat(np.asarray(f), scale, 0), scale, 1) for f in c] for c in clips]
    clips = [[caption(f, lab, where="top") for f in c] for c, lab in zip(clips, labels)]
    T = max(len(c) for c in clips)
    clips = [list(c) + [c[-1]] * (T - len(c) + 10) for c in clips]
    frames = [np.hstack([c[t] for c in clips]) for t in range(T + 10)]
    if footer:
        frames = [caption(f, footer, size=max(12, frames[0].shape[0] // 24)) for f in frames]
    show_video(frames, fps=fps, title=title)
    if export:
        os.makedirs("outputs", exist_ok=True)
        _quiet_export(frames, f"outputs/{_slug(title)}.mp4", fps)
    return frames
