"""Capstone C8: share-ready 1920 x 1080 driving videos.

    ┌──────────────────────────────── 1280 ─┬──────── 640 ──┐
    │ chase camera, the model's 6 sampled   │ bird's-eye map │
    │ plans drawn on the road, the chosen   │ (heading up)   │
    │ one highlighted                  720  │           640  │
    ├──────────────┬────────────────────────┼───────────────┤
    │ front camera │ instruction · reasoning│ speed, plans,  │
    │ (model input)│ + reason probabilities │ scenario stats │
    └──────────────┴────────────────────────┴───────────────┘
"""
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from . import showcase as sc
from .city import COMMANDS, REASON_TEXT, REASONS, DT
from .drive_policy import plan_path

W, HGT = 1920, 1080
PLAN_RGBA = np.array([0.35, 0.8, 1.0, 0.55], np.float32)
CHOSEN_RGBA = np.array([0.2, 1.0, 0.55, 0.95], np.float32)
ROUTE_RGBA = np.array([0.25, 0.55, 1.0, 0.35], np.float32)
TRAIL_RGBA = np.array([1.0, 0.85, 0.2, 0.8], np.float32)
REASON_COL = {"cruise": sc.GOOD, "turn": sc.ACCENT, "car": (255, 200, 80), "red": sc.BAD, "ped": (255, 120, 200),
              "yield": (255, 170, 60)}


def _line(scene, a, b, width, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
    scene.ngeom += 1


def _polyline(scene, pts, z, width, rgba, step=1):
    for p, q in zip(pts[:-step:step], pts[step::step]):
        _line(scene, np.r_[p, z], np.r_[q, z], width, rgba)


def fast_forward(frames, still, min_len=45, factor=4):
    """Waiting at a light is boring on video: play standing-still stretches longer than 1.5 s at `factor`× speed,
    with a badge, so the viewer knows time was compressed."""
    out, i = [], 0
    while i < len(frames):
        j = i
        while j < len(frames) and still[j]:
            j += 1
        if j - i >= min_len:
            out += frames[i:i + 15] + [sc.badge(f, f"▶▶ waiting · ×{factor}", corner="tr", size=24) for f in frames[i + 15:j - 15:factor]] \
                   + frames[j - 15:j]
            i = j
        else:
            out.append(frames[i]); i += 1
    return out


def preview(frames, step=2):
    """Half-resolution, every `step`-th frame: small enough to embed in a notebook."""
    return [np.ascontiguousarray(f[::2, ::2]) for f in frames[::step]]


class Film:
    def __init__(self, env, policy):
        self.env, self.pol = env, policy
        m = env.model
        self.r_chase = mujoco.Renderer(m, 720, 1280)
        self.r_bev = mujoco.Renderer(m, 640, 640)
        self.r_dash = mujoco.Renderer(m, 360, 640)
        self.cam = mujoco.MjvCamera()
        self.bev = mujoco.MjvCamera()
        self.az = None
        self.trail = []

    # -------------------------------------------------------------- overlays
    def _overlays(self, scene, bev=False):
        e, info = self.env.ego, self.pol.last
        z = 0.3 if not bev else 0.5
        r = self.env.route
        s0 = e["s"]
        ii = (r.s > s0) & (r.s < min(s0 + 60, self.env.route_end))
        if ii.sum() > 2:
            _polyline(scene, r.p[ii], 0.16, 0.35 if bev else 0.12, ROUTE_RGBA, step=2)
        if bev and len(self.trail) > 2:
            _polyline(scene, np.array(self.trail[-300:]), 0.4, 0.25, TRAIL_RGBA, step=3)
        if info is None:
            return
        if info.get("paths") is not None:                 # trajectories already in world coordinates (Alpamayo)
            paths = info["paths"]
        elif info.get("plans") is not None:               # (a, κ) plans: integrate with the unicycle model
            paths = [plan_path(e["xy"], e["yaw"], e["v"], plan) for plan in info["plans"]]
        else:
            return
        for k, pts in enumerate(paths):
            chosen = k == info["chosen"]
            _polyline(scene, pts, z + (0.05 if chosen else 0), (0.22 if chosen else 0.08) * (2.2 if bev else 1),
                      CHOSEN_RGBA if chosen else PLAN_RGBA)

    # -------------------------------------------------------------- views
    def _chase(self):
        e = self.env.ego
        yaw = np.degrees(e["yaw"])
        self.az = yaw if self.az is None else self.az + 0.06 * ((yaw - self.az + 180) % 360 - 180)
        f = np.array([np.cos(e["yaw"]), np.sin(e["yaw"])])
        self.cam.lookat[:] = [*(e["xy"] + 5.0 * f), 1.2]
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 14.0, -17.0, self.az
        self.r_chase.update_scene(self.env.data, self.cam)
        self._overlays(self.r_chase.scene)
        return self.r_chase.render()

    def _bird(self):
        e = self.env.ego
        f = np.array([np.cos(e["yaw"]), np.sin(e["yaw"])])
        self.bev.lookat[:] = [*(e["xy"] + 12.0 * f), 0]
        self.bev.distance, self.bev.elevation, self.bev.azimuth = 62.0, -89.9, self.az
        self.r_bev.update_scene(self.env.data, self.bev)
        self._overlays(self.r_bev.scene, bev=True)
        return self.r_bev.render()

    def _dash(self):
        self.r_dash.update_scene(self.env.data, "front")
        self._overlays(self.r_dash.scene)
        return self.r_dash.render()

    # -------------------------------------------------------------- panels
    def _reason_panel(self, size=(360, 640), title=""):
        Hh, Ww = size
        im = Image.new("RGB", (Ww, Hh), sc.BG); d = ImageDraw.Draw(im)
        env, info = self.env, self.pol.last
        d.text((24, 16), "INSTRUCTION", font=sc.font(18, True), fill=sc.ACCENT)
        ins = f"“{(info or {}).get('nav') or env.instruction}”"      # Alpamayo gets its own format ("Turn left in 20m")
        d.text((24, 44), ins, font=sc.fitted(d, ins, Ww - 48, 30, True), fill=sc.FG)
        mode = (info or {}).get("mode")
        who = {"warmup": "WARM-UP: COLLECTING 2 S OF CAMERA HISTORY", "alpamayo": "ALPAMAYO'S REASONING (CANDIDATE 1)",
               "shadow": "ALPAMAYO, SHADOW MODE: WHAT IT WOULD DO"}.get(
            mode, "EXPERT'S REASONING" if info is not None and info.get("plans") is None else "MODEL'S REASONING")
        d.text((24, 104), who, font=sc.font(18, True), fill=sc.ACCENT)
        if info is not None and info.get("text"):
            import textwrap
            f = sc.font(22)
            text = info["text"] if info["text"] != "<NOMINAL>" else "<NOMINAL>  (adaptive reasoning: routine driving, no explanation needed)"
            lines = textwrap.wrap(text, width=50)[:8]
            for k, line in enumerate(lines):
                d.text((24, 136 + k * 28), line, font=f, fill=sc.FG)
            if info.get("latency"):
                d.text((24, Hh - 34), f"model call: {info['latency']:.2f} s on one GPU (the simulation waits)",
                       font=sc.font(16), fill=sc.MUTED)
        elif info is not None and info.get("reason") is not None:
            reason = info["reason"]
            txt = REASON_TEXT[reason]
            d.text((24, 132), txt, font=sc.fitted(d, txt, Ww - 48, 30, True), fill=REASON_COL[reason])
            why = info.get("why")
            if why is not None:
                y0, bw = 186, (Ww - 48)
                for k, rname in enumerate(REASONS):
                    y = y0 + k * 27
                    d.text((24, y), rname, font=sc.font(17), fill=sc.MUTED)
                    x0 = 110
                    d.rectangle([x0, y + 4, x0 + (bw - 120), y + 20], fill=(34, 38, 50))
                    d.rectangle([x0, y + 4, x0 + int((bw - 120) * float(why[k])), y + 20], fill=REASON_COL[rname])
                    d.text((x0 + bw - 110, y), f"{why[k]:.0%}", font=sc.font(17), fill=sc.FG)
        return np.asarray(im)

    def _stats_panel(self, size=(440, 640), scenario=""):
        Hh, Ww = size
        im = Image.new("RGB", (Ww, Hh), sc.BG); d = ImageDraw.Draw(im)
        e, env = self.env.ego, self.env
        d.text((24, 18), "SPEED", font=sc.font(18, True), fill=sc.ACCENT)
        d.text((24, 44), f"{e['v'] * 3.6:4.0f} km/h", font=sc.font(54, True), fill=sc.FG)
        frac = min(1.0, e["v"] / 12.0)
        d.rectangle([24, 116, Ww - 24, 128], fill=(34, 38, 50)); d.rectangle([24, 116, 24 + int((Ww - 48) * frac), 128], fill=sc.ACCENT)
        d.text((24, 150), "SCENARIO", font=sc.font(18, True), fill=sc.ACCENT)
        d.text((24, 176), scenario, font=sc.fitted(d, scenario, Ww - 48, 28, True), fill=sc.FG)
        d.text((24, 226), f"time {env.t:5.1f} s    distance {env.dist:5.0f} m", font=sc.font(22), fill=sc.MUTED)
        ok = not (env.events["collision"] or env.events["offroad"])
        d.text((24, 262), "incidents: none" if ok else f"incident: {env.events['collision'] or 'off road'}",
               font=sc.font(22, True), fill=sc.GOOD if ok else sc.BAD)
        y = 300
        d.text((24, y), "LEGEND", font=sc.font(18, True), fill=sc.ACCENT)
        for k, (col, txt) in enumerate([(tuple(int(255 * c) for c in PLAN_RGBA[:3]), "6 sampled plans (flow matching)"),
                                        (tuple(int(255 * c) for c in CHOSEN_RGBA[:3]),
                                         "candidate 1 (not driven: expert drives)" if (self.pol.last or {}).get("mode") == "shadow"
                                         else "the plan being driven"),
                                        (tuple(int(255 * c) for c in ROUTE_RGBA[:3]), "navigation route"),
                                        (tuple(int(255 * c) for c in TRAIL_RGBA[:3]), "path driven so far (map)")]):
            yy = y + 28 + k * 25
            d.rectangle([24, yy + 8, 60, yy + 14], fill=col)
            d.text((72, yy), txt, font=sc.font(18), fill=sc.FG)
        return np.asarray(im)

    def frame(self, scenario="", title=""):
        e = self.env.ego
        if not self.trail or np.linalg.norm(self.trail[-1] - e["xy"]) > 0.4:
            self.trail.append(e["xy"].copy())
        chase = sc.caption(self._chase(), title, where="top", size=26)
        bev = sc.badge(self._bird(), "bird's-eye view", corner="tl", size=18)
        dash = sc.badge(self._dash(), "front camera · the model sees it at 224×128", corner="tl", size=15)
        canvas = np.empty((HGT, W, 3), np.uint8)
        canvas[:720, :1280] = chase
        canvas[:640, 1280:] = bev
        canvas[720:, :640] = dash
        canvas[720:, 640:1280] = self._reason_panel()
        canvas[640:, 1280:] = self._stats_panel(scenario=scenario)
        canvas[718:722, :1280] = sc.BG; canvas[:, 1278:1282] = sc.BG; canvas[638:642, 1280:] = sc.BG; canvas[720:, 638:642] = sc.BG
        return canvas

    # -------------------------------------------------------------- an episode
    def episode(self, seed, scenario=None, label="", title="", max_seconds=None):
        from .drive_policy import run_episode
        frames, still = [], []
        self.az, self.trail = None, []

        def grab(env):
            frames.append(self.frame(scenario=label, title=title)); still.append(env.ego["v"] < 0.15)
        self.env._substep_cb = grab
        try:
            info = run_episode(self.env, self.pol, seed, scenario)
        finally:
            self.env._substep_cb = None
        frames = fast_forward(frames, still)
        ok = info["success"]
        end = sc.badge(frames[-1], "✓ arrived" if ok else "✗ " + (str(info["collision"] or "") or "failed"),
                       corner="bl", size=34, color=sc.GOOD if ok else sc.BAD)
        return frames + [end] * 45, info
