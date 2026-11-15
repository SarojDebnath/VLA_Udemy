"""Capstone C8: autonomous driving in a procedural MuJoCo city.

    City        a 4 x 4-intersection city built with MjSpec: roads, lane markings, crosswalks, traffic lights,
                buildings, parks and trees. The ego car, the traffic cars and the pedestrians are mocap bodies moved
                by the simulation below (kinematic vehicles, as in most driving simulators' traffic models).
    Lights      fixed-cycle traffic lights, a random phase per intersection
    Traffic     cars on random routes with IDM car following (they stop for red lights, pedestrians and each other);
                pedestrians walk the sidewalks and cross at crosswalks on the walk signal; an optional jaywalker
                steps into the road in front of the ego car
    Expert      the privileged driver that records the demonstrations: pure pursuit on the route + IDM speed control
                that stops for red lights, pedestrians and the car ahead, slows for turns, and yields to oncoming
                traffic before turning left. It also says WHY (the label of the policy's reasoning head)
    DriveEnv    an episode: a route through two intersections, one navigation instruction per intersection
                ("turn left at the next intersection"), 10 Hz control of (acceleration, curvature)
    Film        share-ready 1920 x 1080 video: chase camera, dash camera, bird's-eye map, instruction, reasoning

Right-hand traffic. Units: metres, seconds, radians. Control: a in m/s^2, kappa (curvature) in 1/m.
"""
import json
import os
import time

import mujoco
import numpy as np

# ---------------------------------------------------------------- city geometry
N = 4                       # intersections per side
PITCH = 60.0                # distance between intersections
RW = 5.0                    # road half-width (two lanes)
LANE = 1.75                 # lane centre offset from the road centre line
SIDEWALK = 8.0              # sidewalk outer edge (from the road centre line)
CORNER = 6.6                # where pedestrians walk / cross (inside the crosswalk band)
STOP = 8.8                  # stop line, from the intersection centre
CAR_L, CAR_W = 4.4, 1.9
DT = 0.1                    # control period (10 Hz, like Alpamayo's trajectory)
SUB = 3                     # simulation substeps per control step (30 Hz video)
A_MIN, A_MAX = -7.0, 3.0    # ego acceleration limits
K_MAX, K_RATE = 0.33, 1.0   # curvature limit (5 m turning radius) and rate (1/m per s)

D = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], float)          # E, N, W, S
RIGHT = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]], float)      # right-hand side of each direction
DIR_NAME = ["east", "north", "west", "south"]

# traffic lights: green, yellow, all-red, per axis (0 = east-west traffic, 1 = north-south traffic)
G_T, Y_T, AR_T = 12.0, 3.0, 2.0
CYCLE = 2 * (G_T + Y_T + AR_T)

# IDM car following
IDM = dict(a=2.0, b=3.0, s0=2.0, T=1.2)
A_LAT = 2.5                 # comfortable lateral acceleration in turns (m/s^2)

REASONS = ["cruise", "turn", "car", "red", "ped", "yield"]
REASON_TEXT = {
    "cruise": "road is clear: keeping speed",
    "turn": "slowing down for the turn",
    "car": "following the car ahead",
    "red": "red light: stopping at the line",
    "ped": "pedestrian in the way: braking",
    "yield": "turning left: waiting for oncoming traffic",
}
COMMANDS = {"L": "turn left at the next intersection", "S": "go straight at the next intersection",
            "R": "turn right at the next intersection", "F": "follow the road"}
PARAPHRASES = {
    "L": ["turn left at the next intersection", "take the next left", "make a left at the intersection ahead",
          "go left at the next junction"],
    "S": ["go straight at the next intersection", "continue straight through the intersection",
          "keep going straight at the next junction", "drive straight ahead through the crossing"],
    "R": ["turn right at the next intersection", "take the next right", "make a right at the intersection ahead",
          "go right at the next junction"],
    "F": ["follow the road", "keep driving along this road", "stay on this road", "continue down the road"],
}


def node_xy(node):
    return np.array([node[0] * PITCH, node[1] * PITCH], float)


def in_grid(node):
    return 0 <= node[0] < N and 0 <= node[1] < N


def step_node(node, d):
    return (node[0] + int(D[d][0]), node[1] + int(D[d][1]))


def lane_pt(node, d, along):
    """Point on the lane of travel direction d, `along` metres from the intersection centre (negative = before it)."""
    return node_xy(node) + D[d] * along + RIGHT[d] * LANE


def turn_type(d_in, d_out):
    return "S" if d_in == d_out else ("L" if d_out == (d_in + 1) % 4 else "R")


def turn_dir(d_in, t):
    return {"S": d_in, "L": (d_in + 1) % 4, "R": (d_in + 3) % 4}[t]


def _bezier(p0, t0, p3, t3, n=24):
    k = 0.55 * np.linalg.norm(p3 - p0) / np.sqrt(2)
    p1, p2 = p0 + t0 * k, p3 - t3 * k
    u = np.linspace(0, 1, n)[:, None]
    return (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * p1 + 3 * (1 - u) * u ** 2 * p2 + u ** 3 * p3


class Path:
    """A lane-centre polyline resampled every 0.5 m: positions, arc length s, heading, curvature, and the junctions
    it goes through (node, entry/exit direction, turn, s of the stop line / junction entry / junction exit)."""
    DS = 0.5

    def __init__(self, start_node, d, along0, dirs, end_along):
        pieces, juncs, node = [], [], start_node
        a = lane_pt(node, d, along0)
        for k, d2 in enumerate(list(dirs) + [None]):
            nb = step_node(node, d)
            if d2 is None:
                pieces.append(np.stack([a, lane_pt(nb, d, end_along) if end_along is not None else lane_pt(nb, d, -RW)]))
                break
            b = lane_pt(nb, d, -RW)
            pieces.append(np.stack([a, b]))
            j_pts = (np.stack([b, lane_pt(nb, d2, RW)]) if d2 == d
                     else _bezier(b, D[d], lane_pt(nb, d2, RW), D[d2]))
            juncs.append(dict(node=nb, d_in=d, d_out=d2, turn=turn_type(d, d2), piece=len(pieces)))
            pieces.append(j_pts)
            node, d, a = nb, d2, lane_pt(nb, d2, RW)
        raw, marks = [pieces[0]], [0]
        for pc in pieces[1:]:
            marks.append(sum(len(r) for r in raw) - 1)
            raw.append(pc[1:])
        raw = np.concatenate(raw)
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(raw, axis=0), axis=1))]
        self.length = seg[-1]
        self.s = np.arange(0, self.length, self.DS)
        self.p = np.stack([np.interp(self.s, seg, raw[:, 0]), np.interp(self.s, seg, raw[:, 1])], 1)
        dp = np.gradient(self.p, axis=0)
        self.yaw = np.unwrap(np.arctan2(dp[:, 1], dp[:, 0]))
        self.kappa = np.gradient(self.yaw) / self.DS
        self.junctions = []
        for J in juncs:
            s_in = seg[marks[J["piece"]]]
            s_out = seg[marks[J["piece"] + 1]] if J["piece"] + 1 < len(marks) else self.length
            self.junctions.append(dict(J, s_in=s_in, s_out=s_out, s_stop=s_in - (STOP - RW)))

    def idx(self, s):
        i = int(s / self.DS + 0.5)
        return 0 if i < 0 else (len(self.s) - 1 if i >= len(self.s) else i)

    def pose(self, s):
        i = self.idx(s)
        return self.p[i], self.yaw[i]

    def project(self, xy, s_guess, window=30.0):
        """Arc length of the path point nearest to xy, searched around s_guess; and the signed lateral offset."""
        i0, i1 = self.idx(s_guess - window), self.idx(s_guess + window) + 1
        d = np.linalg.norm(self.p[i0:i1] - xy, axis=1)
        i = i0 + int(np.argmin(d))
        n = np.array([-np.sin(self.yaw[i]), np.cos(self.yaw[i])])
        return self.s[i], float(np.dot(xy - self.p[i], n))

    def next_junction(self, s, margin=1.0):
        for J in self.junctions:
            if s < J["s_out"] - margin:
                return J
        return None


def random_dirs(node, d, n, rng, allow_left=True):
    """n exit directions for a route starting on the road leaving `node` in direction d. Never a U-turn.
    allow_left=False forbids left turns that cross an oncoming lane (traffic cars), unless it's the only way."""
    out = []
    for _ in range(n):
        nb = step_node(node, d)
        opts = [d2 for d2 in range(4) if d2 != (d + 2) % 4 and in_grid(step_node(nb, d2))]
        if not allow_left:
            onc = in_grid(step_node(nb, d))           # the road straight ahead exists: oncoming traffic comes from it
            safe = [d2 for d2 in opts if not (turn_type(d, d2) == "L" and onc)]
            opts = safe or opts
        w = np.array([3.0 if d2 == d else 1.5 for d2 in opts])
        d2 = opts[rng.choice(len(opts), p=w / w.sum())]
        out.append(d2)
        node, d = nb, d2
    return out


# ---------------------------------------------------------------- traffic lights
class Lights:
    def __init__(self, rng):
        self.off = {(i, j): rng.uniform(0, CYCLE) for i in range(N) for j in range(N)}

    def _u(self, node, axis, t):
        u = (t + self.off[node]) % CYCLE
        return u if axis == 0 else (u - (G_T + Y_T + AR_T)) % CYCLE

    def state(self, node, d, t):
        u = self._u(node, d % 2, t)
        return "G" if u < G_T else ("Y" if u < G_T + Y_T else "R")

    def green_left(self, node, axis, t):
        u = self._u(node, axis, t)
        return G_T - u if u < G_T else -1.0


# ---------------------------------------------------------------- pedestrians
def corner_xy(c):
    return node_xy(c[:2]) + np.array([c[2], c[3]], float) * CORNER


def corner_links(c):
    """(next corner, kind, axis that must be green to walk) for sidewalk corner c = (i, j, qx, qy)."""
    i, j, qx, qy = c
    out = []
    if in_grid((i, j + qy)):                                  # crosswalk over the north/south arm: walk on east-west green
        out.append(((i, j, -qx, qy), "cross", 0))
        out.append(((i, j + qy, qx, -qy), "walk", None))
    if in_grid((i + qx, j)):                                  # crosswalk over the east/west arm: walk on north-south green
        out.append(((i, j, qx, -qy), "cross", 1))
        out.append(((i + qx, j, -qx, qy), "walk", None))
    return out


PED_COLORS = [(0.85, 0.25, 0.25), (0.2, 0.45, 0.85), (0.95, 0.75, 0.15), (0.25, 0.7, 0.4), (0.6, 0.35, 0.75),
              (0.95, 0.5, 0.2), (0.15, 0.7, 0.75), (0.9, 0.9, 0.9), (0.3, 0.3, 0.35)]


def _blocked(pos, direction, cars_xy, cars_v, r=2.8):
    """People don't walk into the side of a moving car: is a moving car within r of the point 1 m ahead?"""
    if cars_xy is None or not len(cars_xy):
        return False
    q = pos + direction * 1.0
    near = np.linalg.norm(cars_xy - q, axis=1) < r
    return bool(np.any(near & (cars_v > 0.3)))


class Ped:
    def __init__(self, rng):
        while True:
            c = (int(rng.integers(N)), int(rng.integers(N)), int(rng.choice([-1, 1])), int(rng.choice([-1, 1])))
            if corner_links(c):
                break
        walks = [l for l in corner_links(c) if l[1] == "walk"] or corner_links(c)
        nxt = walks[rng.integers(len(walks))]
        f = rng.uniform(0.05, 0.95)
        self.prev, self.target, self.kind, self.axis = c, nxt[0], nxt[1], nxt[2]
        self.pos = (1 - f) * corner_xy(c) + f * corner_xy(nxt[0])
        self.speed = rng.uniform(1.15, 1.6)
        self.waiting, self.phase, self.rng = False, rng.uniform(0, 6.28), rng
        self.vel = np.zeros(2)

    def update(self, h, t, lights, cars_xy=None, cars_v=None):
        goal = corner_xy(self.target)
        if self.waiting:                                      # at the kerb, waiting for the walk signal
            if lights.green_left(self.target[:2], self.axis, t) > 8.0:
                self.waiting = False
            else:
                self.vel[:] = 0
                return
        d = goal - self.pos
        dist = np.linalg.norm(d)
        if dist < 0.05:
            self._choose_next()
            return
        if _blocked(self.pos, d / dist, cars_xy, cars_v):                # a moving car right ahead: let it pass
            self.vel[:] = 0
            return
        step = min(self.speed * h, dist)
        self.vel = d / dist * self.speed
        self.pos = self.pos + d / dist * step
        self.phase += 7.0 * h

    def _choose_next(self):
        here = self.target
        opts = [l for l in corner_links(here) if l[0] != self.prev] or corner_links(here)
        nxt = opts[self.rng.integers(len(opts))]
        self.prev, self.target, self.kind, self.axis = here, nxt[0], nxt[1], nxt[2]
        self.pos = corner_xy(here)
        self.waiting = nxt[1] == "cross"


class Jaywalker:
    """Steps off the kerb on the ego car's right and crosses the road mid-block when the ego car gets close."""

    def __init__(self, path, s_cross, rng):
        p, yaw = path.pose(s_cross)
        n = np.array([-np.sin(yaw), np.cos(yaw)])               # left of the ego lane
        road_c = p + n * LANE                                    # road centre line
        self.start, self.end = road_c - n * (RW + 1.4), road_c + n * (RW + 1.4)
        self.pos, self.s_cross = self.start.copy(), s_cross
        self.speed = rng.uniform(1.5, 1.9)
        self.trigger_lead = rng.uniform(3.2, 4.0)                 # starts when the ego car is ~this many seconds away
        self.walking, self.done, self.phase, self.vel = False, False, 0.0, np.zeros(2)

    def update(self, h, ego_s, ego_v, cars_xy=None, cars_v=None):
        if not self.walking and not self.done and self.s_cross - ego_s < max(ego_v, 5.0) * self.trigger_lead + 4.0:
            self.walking = True
        if self.walking:
            d = self.end - self.pos
            dist = np.linalg.norm(d)
            if dist < 0.05:
                self.walking, self.done = False, True
                self.vel[:] = 0
                return
            if _blocked(self.pos, d / dist, cars_xy, cars_v, r=2.4):
                self.vel[:] = 0
                return
            self.vel = d / dist * self.speed
            self.pos = self.pos + d / dist * min(self.speed * h, dist)
            self.phase += 7.0 * h


# ---------------------------------------------------------------- vehicles
class Car:
    """A traffic car: follows its path with IDM."""

    def __init__(self, path, s, v, v_des, color):
        self.path, self.v, self.v_des, self.color, self.a = path, v, v_des, color, 0.0
        self.move_to(s)

    def move_to(self, s):
        self.s = min(s, self.path.length - 1)
        i = self.path.idx(self.s)
        self.xy, self.yaw = self.path.p[i], float(self.path.yaw[i])

    @property
    def vel(self):
        return self.v * np.array([np.cos(self.yaw), np.sin(self.yaw)])


def idm_accel(v, v_lim, cands):
    """IDM: free-road term toward v_lim plus the strongest interaction among obstacle candidates (gap, v_obstacle, kind).
    Returns (acceleration, reason)."""
    a, b, s0, T = IDM["a"], IDM["b"], IDM["s0"], IDM["T"]
    free = a * (1 - (v / max(v_lim, 0.1)) ** 4)
    worst, kind = 0.0, None
    for gap, vo, k in cands:
        sstar = s0 + max(0.0, v * T + v * (v - vo) / (2 * np.sqrt(a * b)))
        inter = a * (sstar / max(gap, 0.2)) ** 2
        if inter > worst:
            worst, kind = inter, k
    acc = free - worst
    return float(np.clip(acc, -8.0, a)), kind, worst


def longitudinal(path, s, v, v_des, t, lights, others_xy, others_vel, peds_xy, oncoming_fn=None):
    """Speed control for a vehicle at arc length s: curvature speed limit, obstacles in the corridor ahead,
    traffic lights, and (if oncoming_fn) yielding before a left turn. Returns (acceleration, reason)."""
    i0 = path.idx(s)
    ahead = slice(i0, min(i0 + 120, len(path.s)))
    ds = path.s[ahead] - s
    k = np.abs(path.kappa[ahead])
    v_turn = np.sqrt(A_LAT / np.maximum(k, 1e-3))
    v_allow = np.sqrt(v_turn ** 2 + 2 * 2.0 * np.maximum(ds - 1.0, 0))
    v_lim = min(v_des, float(v_allow.min()))
    cands = []
    look = np.arange(i0 + 4, min(i0 + 100, len(path.s)), 2)
    if len(look):
        P, s_look = path.p[look], path.s[look]
        tang = np.stack([np.cos(path.yaw[look]), np.sin(path.yaw[look])], 1)
        if len(others_xy):
            dd = np.linalg.norm(P[:, None] - others_xy[None], axis=2)              # (K, M)
            hit = dd < 1.9
            for m in np.where(hit.any(0))[0]:
                kk = int(np.argmax(hit[:, m]))
                cands.append((s_look[kk] - s - CAR_L, max(0.0, float(others_vel[m] @ tang[kk])), "car"))
        front = s_look > s + CAR_L / 2 + 0.3                      # people beside the car body are not in its way
        if len(peds_xy) and front.any():
            dd = np.linalg.norm(P[front][:, None] - peds_xy[None], axis=2)
            hit = dd < 2.1
            for m in np.where(hit.any(0))[0]:
                kk = int(np.argmax(hit[:, m]))
                cands.append((s_look[front][kk] - s - CAR_L / 2 - 0.8, 0.0, "ped"))
    J = path.next_junction(s)
    if J is not None and J["s_stop"] - s < 80:
        front = s + CAR_L / 2
        dist = J["s_stop"] - front
        st = lights.state(J["node"], J["d_in"], t)
        if dist > -0.5 and (st == "R" or (st == "Y" and dist > v * v / (2 * 4.0) + 0.5)):
            cands.append((dist - 1.0 + IDM["s0"], 0.0, "red"))              # stop ~1 m before the line
        elif oncoming_fn is not None and J["turn"] == "L" and dist > -0.5 and oncoming_fn(J):
            cands.append((dist - 1.0 + IDM["s0"], 0.0, "yield"))
    acc, kind, worst = idm_accel(v, v_lim, cands)
    if kind is not None and (worst > 0.4 or v < 0.5):
        reason = kind
    else:
        reason = "turn" if v_lim < v_des - 0.6 else "cruise"
    return acc, reason


def pure_pursuit(path, s, xy, yaw, v):
    ld = float(np.clip(0.6 * v + 3.0, 3.5, 9.0))
    p, _ = path.pose(s + ld)
    d = p - xy
    alpha = np.arctan2(d[1], d[0]) - yaw
    alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
    return float(np.clip(2 * np.sin(alpha) / max(np.linalg.norm(d), 1.0), -K_MAX, K_MAX))


def car_circles(xy, yaw):
    f = np.array([np.cos(yaw), np.sin(yaw)])
    return np.stack([xy + f * o for o in (-1.35, 0.0, 1.35)])


def on_road(xy):
    gx, gy = xy / PITCH
    lo, hi = -RW - 0.4, (N - 1) * PITCH + RW + 0.4
    if not (lo <= xy[0] <= hi and lo <= xy[1] <= hi):
        return False
    return abs(xy[0] - round(gx) * PITCH) <= RW + 0.4 or abs(xy[1] - round(gy) * PITCH) <= RW + 0.4


# ---------------------------------------------------------------- the MuJoCo city
def _hex(c):
    return [c[0], c[1], c[2], 1.0]


def lookat_xyaxes(pos, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, up); x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return np.concatenate([x, y]).tolist()


CAR_COLORS = [(0.75, 0.12, 0.12), (0.1, 0.25, 0.6), (0.15, 0.15, 0.17), (0.8, 0.8, 0.82), (0.55, 0.57, 0.6),
              (0.1, 0.45, 0.35), (0.85, 0.55, 0.1), (0.35, 0.2, 0.45), (0.92, 0.92, 0.92), (0.2, 0.55, 0.8)]

_BOX_FACES = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                       [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
_BOX_CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)


def _icosphere():
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], float)
    f = [[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6],
         [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7],
         [9, 8, 1]]
    v = list(v / np.linalg.norm(v, axis=1, keepdims=True))
    cache, out = {}, []

    def mid(a, b):
        k = (min(a, b), max(a, b))
        if k not in cache:
            m = (v[a] + v[b]) / 2
            v.append(m / np.linalg.norm(m))
            cache[k] = len(v) - 1
        return cache[k]
    for a, b, c in f:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        out += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
    return np.array(v), np.array(out)


_SPH_V, _SPH_F = _icosphere()


class MeshBatch:
    """Many static boxes / cylinders / spheres merged into ONE mesh: one draw call instead of hundreds.
    (MuJoCo draws each geom separately: a city of thousands of geoms renders slowly, a dozen meshes renders fast.)"""

    def __init__(self):
        self.v, self.f, self.n = [], [], 0

    def _add(self, v, f):
        self.v.append(np.asarray(v, float)); self.f.append(np.asarray(f) + self.n); self.n += len(v)

    def box(self, c, half, yaw=0.0):
        cy, sy = np.cos(yaw), np.sin(yaw)
        R = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        self._add(_BOX_CORNERS * np.asarray(half, float) @ R.T + np.asarray(c, float), _BOX_FACES)

    def cylinder(self, c, r, half_h, n=10, axis="z"):
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ring = np.stack([r * np.cos(a), r * np.sin(a)], 1)
        v = np.vstack([np.c_[ring, -np.full(n, half_h)], np.c_[ring, np.full(n, half_h)], [[0, 0, -half_h], [0, 0, half_h]]])
        f = []
        for k in range(n):
            k2 = (k + 1) % n
            f += [[k, k2, n + k2], [k, n + k2, n + k], [2 * n, k2, k], [2 * n + 1, n + k, n + k2]]
        f = np.array(f)
        if axis == "y":                       # swap y and z (a mirror), and flip the faces to keep them outward
            v, f = v[:, [0, 2, 1]], f[:, ::-1]
        self._add(v + np.asarray(c, float), f)

    def sphere(self, c, r):
        self._add(_SPH_V * r + np.asarray(c, float), _SPH_F)

    def mesh(self, spec, name):
        spec.add_mesh(name=name, uservert=np.concatenate(self.v).ravel().tolist(),
                      userface=np.concatenate(self.f).ravel().tolist())
        return name


def build_city(n_cars, n_peds, seed=0, av_rig=False):
    """Compile the city. The building layout is fixed by `seed`; cars and people are mocap bodies placed every step."""
    rng = np.random.default_rng(seed)
    spec = mujoco.MjSpec()
    spec.option.timestep = DT / SUB
    spec.option.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT) | int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.visual.quality.shadowsize = 8192
    spec.visual.map.znear, spec.visual.map.zfar = 0.002, 30.0          # relative to the extent (120 m)
    spec.visual.map.fogstart, spec.visual.map.fogend = 0.8, 3.0
    spec.visual.map.shadowclip = 0.3                                    # no shadow acne on the long streets
    spec.visual.rgba.fog = [0.78, 0.84, 0.9, 1]
    spec.visual.rgba.haze = [0.8, 0.86, 0.93, 1]
    spec.visual.headlight.ambient = [0.42, 0.42, 0.45]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    spec.visual.headlight.specular = [0.05, 0.05, 0.05]
    spec.stat.extent = 120.0
    spec.stat.center = [PITCH * (N - 1) / 2, PITCH * (N - 1) / 2, 0]
    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.45, 0.62, 0.85], rgb2=[0.92, 0.94, 0.97], width=512, height=512)

    def material(name, rgb, refl=0.0, emission=0.0, spec_=0.3):
        spec.add_material(name=name, rgba=_hex(rgb), reflectance=refl, emission=emission, specular=spec_)
        return name
    for name, rgb in [("ground", (0.36, 0.42, 0.33)), ("asphalt", (0.2, 0.21, 0.23)), ("sidewalk", (0.62, 0.62, 0.6)),
                      ("white", (0.95, 0.95, 0.95)), ("yellow", (0.95, 0.75, 0.15)), ("grass", (0.3, 0.5, 0.25)),
                      ("trunk", (0.35, 0.25, 0.17)), ("leaves", (0.2, 0.42, 0.2)), ("pole", (0.18, 0.19, 0.2)),
                      ("tyre", (0.06, 0.06, 0.06))]:
        material(name, rgb, spec_=0.05)
    material("glass", (0.16, 0.22, 0.3), refl=0.2, spec_=0.8)
    material("lamp", (1, 1, 1), emission=0.9)
    material("dimlamp", (0.12, 0.12, 0.1), spec_=0.5)
    material("headlight", (1, 0.97, 0.85), emission=0.6)
    material("taillight", (0.8, 0.05, 0.05), emission=0.4)
    facade_rgb = [(0.82, 0.78, 0.7), (0.7, 0.72, 0.76), (0.62, 0.52, 0.45), (0.85, 0.84, 0.82), (0.55, 0.6, 0.66), (0.76, 0.66, 0.55)]
    facades = [material(f"facade{k}", c) for k, c in enumerate(facade_rgb)]
    wb = spec.worldbody
    wb.add_light(name="sun", type=int(mujoco.mjtLightType.mjLIGHT_DIRECTIONAL), castshadow=True, pos=[90, 60, 120],
                 dir=[-0.35, -0.25, -0.9], diffuse=[0.75, 0.73, 0.68], specular=[0.2, 0.2, 0.2])
    span = (N - 1) * PITCH
    c0 = span / 2
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[400, 400, 1], pos=[c0, c0, 0], material="ground")
    B = {k: MeshBatch() for k in ["sidewalk", "asphalt", "white", "yellow", "grass", "glass", "trunk", "leaves", "pole",
                                  "dimlamp"] + facades}
    for k in range(N):                                   # sidewalks, then asphalt on top
        v = k * PITCH
        B["sidewalk"].box([c0, v, 0.06], [c0 + SIDEWALK, SIDEWALK, 0.06])
        B["sidewalk"].box([v, c0, 0.06], [SIDEWALK, c0 + SIDEWALK, 0.06])
        B["asphalt"].box([c0, v, 0.125], [c0 + RW, RW, 0.01])
        B["asphalt"].box([v, c0, 0.126], [RW, c0 + RW, 0.01])
    z_mark = 0.14
    for k in range(N):                                   # centre lines (double yellow) and edge lines
        v = k * PITCH
        for a in range(N - 1):
            u0, u1 = a * PITCH + STOP, (a + 1) * PITCH - STOP
            um, uh = (u0 + u1) / 2, (u1 - u0) / 2
            for off in (-0.12, 0.12):
                B["yellow"].box([um, v + off, z_mark], [uh, 0.06, 0.004])
                B["yellow"].box([v + off, um, z_mark], [0.06, uh, 0.004])
            for side in (-1, 1):
                B["white"].box([um, v + side * (RW - 0.25), z_mark], [uh + STOP - RW, 0.07, 0.004])
                B["white"].box([v + side * (RW - 0.25), um, z_mark], [0.07, uh + STOP - RW, 0.004])
    for i in range(N):                                   # crosswalks and stop lines
        for j in range(N):
            c = node_xy((i, j))
            for d in range(4):
                if not in_grid(step_node((i, j), d)):
                    continue
                f, r = D[d], RIGHT[d]
                for w in np.arange(-RW + 0.6, RW - 0.4, 1.1):
                    p = c + f * CORNER + r * w
                    B["white"].box([p[0], p[1], z_mark], [0.6, 0.28, 0.004] if d % 2 == 0 else [0.28, 0.6, 0.004])
                din = (d + 2) % 4
                p = c + f * STOP + RIGHT[din] * (RW / 2)
                B["white"].box([p[0], p[1], z_mark], [0.22, RW / 2, 0.004] if d % 2 == 0 else [RW / 2, 0.22, 0.004])
    parks = {(1, 1), (-1, 2)}
    for bi in range(-1, N):                              # buildings and parks (with a ring of blocks around the grid)
        for bj in range(-1, N):
            x0, x1 = bi * PITCH + SIDEWALK + 1.0, (bi + 1) * PITCH - SIDEWALK - 1.0
            y0, y1 = bj * PITCH + SIDEWALK + 1.0, (bj + 1) * PITCH - SIDEWALK - 1.0
            if (bi, bj) in parks:
                B["grass"].box([(x0 + x1) / 2, (y0 + y1) / 2, 0.08], [(x1 - x0) / 2 + 1, (y1 - y0) / 2 + 1, 0.08])
                for _ in range(14):
                    tx, ty, hgt = rng.uniform(x0 + 2, x1 - 2), rng.uniform(y0 + 2, y1 - 2), rng.uniform(4, 7)
                    B["trunk"].cylinder([tx, ty, hgt / 2], 0.25, hgt / 2)
                    B["leaves"].sphere([tx, ty, hgt + 0.8], rng.uniform(1.8, 2.8))
                continue
            outer = not (0 <= bi < N - 1 and 0 <= bj < N - 1)
            nx, ny = int(rng.integers(2, 4)), int(rng.integers(2, 4))
            xs, ys = np.linspace(x0, x1, nx + 1), np.linspace(y0, y1, ny + 1)
            for a in range(nx):
                for b in range(ny):
                    if 0 < a < nx - 1 and 0 < b < ny - 1:
                        continue
                    g = rng.uniform(0.6, 1.4)
                    bx0, bx1, by0, by1 = xs[a] + g, xs[a + 1] - g, ys[b] + g, ys[b + 1] - g
                    hgt = rng.uniform(14, 48) if not outer else rng.uniform(20, 60)
                    fm = facades[rng.integers(len(facades))]
                    cx, cy, hx, hy = (bx0 + bx1) / 2, (by0 + by1) / 2, (bx1 - bx0) / 2, (by1 - by0) / 2
                    B[fm].box([cx, cy, hgt / 2], [hx, hy, hgt / 2])
                    for fl in np.arange(3.2, hgt - 1.5, 3.4):               # ribbon windows, one band per floor
                        B["glass"].box([cx, cy, fl + 0.9], [hx + 0.04, hy + 0.04, 0.75])
                    B[fm].box([cx, cy, hgt + 0.25], [hx + 0.15, hy + 0.15, 0.25])      # parapet
    for k in range(N):                                   # street trees
        for a in range(N - 1):
            for u in np.arange(a * PITCH + 14, (a + 1) * PITCH - 13, 11.0):
                for side in (-1, 1):
                    if rng.random() < 0.55:
                        continue
                    for (tx, ty) in ((u, k * PITCH + side * (SIDEWALK - 0.9)), (k * PITCH + side * (SIDEWALK - 0.9), u)):
                        B["trunk"].cylinder([tx, ty, 1.7], 0.15, 1.6)
                        B["leaves"].sphere([tx, ty, 3.9], 1.3)
    # traffic lights: one head per approach, on a mast over the lane, at the far right corner. The housing and the
    # three unlit lamps are static mesh; ONE bright lamp geom per head moves to the red / yellow / green position.
    lamps = {}
    for i in range(N):
        for j in range(N):
            c = node_xy((i, j))
            for din in range(4):
                if not in_grid(step_node((i, j), (din + 2) % 4)):
                    continue
                f, r = D[din], RIGHT[din]
                base = c + f * (RW + 1.6) + r * (RW + 1.2)
                head = c + f * (RW + 1.6) + r * LANE
                B["pole"].cylinder([base[0], base[1], 3.1], 0.13, 3.1)
                mid, arm = (base + head) / 2, head - base
                B["pole"].box([mid[0], mid[1], 6.0], [np.linalg.norm(arm) / 2 + 0.1, 0.06, 0.06], yaw=float(np.arctan2(arm[1], arm[0])))
                B["pole"].box([head[0], head[1], 5.0], [0.3, 0.42, 1.05] if din % 2 == 0 else [0.42, 0.3, 1.05])
                p = head - f * 0.31
                for z in LAMP_Z:
                    B["dimlamp"].sphere([p[0], p[1], z], 0.26)
                g = wb.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.28], pos=[p[0] - f[0] * 0.03, p[1] - f[1] * 0.03, LAMP_Z[2]],
                                material="lamp", name=f"lamp_{i}_{j}_{din}", contype=0, conaffinity=0)
                lamps[(i, j, din)] = g.name
    for name, batch in B.items():
        wb.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=batch.mesh(spec, f"m_{name}"), material=name,
                    contype=0, conaffinity=0)
    # vehicles: shared meshes for wheels and lights, boxes for body and cabin (6 geoms per car)
    wheels, heads, tails = MeshBatch(), MeshBatch(), MeshBatch()
    for sx in (1.35, -1.35):
        for sy in (CAR_W / 2 - 0.05, -CAR_W / 2 + 0.05):
            wheels.cylinder([sx, sy, 0.38], 0.36, 0.13, axis="y")
    for sy in (0.62, -0.62):
        heads.box([CAR_L / 2, sy, 0.78], [0.03, 0.2, 0.07])
        tails.box([-CAR_L / 2, sy, 0.8], [0.03, 0.22, 0.07])
    for nm, mb in (("wheels", wheels), ("heads", heads), ("tails", tails)):
        mb.mesh(spec, f"car_{nm}")

    def car_body(name, color, ego=False):
        b = wb.add_body(name=name, mocap=True, pos=[0, 0, 0])
        kw = dict(contype=0, conaffinity=0)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[CAR_L / 2, CAR_W / 2, 0.38], pos=[0, 0, 0.62], rgba=_hex(color), **kw)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[1.15, CAR_W / 2 - 0.08, 0.33], pos=[-0.25, 0, 1.28],
                   rgba=[0.12, 0.15, 0.2, 1], **kw)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[1.05, CAR_W / 2 - 0.06, 0.04], pos=[-0.25, 0, 1.63], rgba=_hex(color), **kw)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="car_wheels", material="tyre", **kw)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="car_heads", material="headlight", **kw)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="car_tails", material="taillight", **kw)
        if ego:                                                     # roof sensor pod + a blue stripe: easy to spot
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.22, 0.12], pos=[-0.1, 0, 1.79], rgba=[0.1, 0.1, 0.12, 1], **kw)
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[1.06, 0.1, 0.02], pos=[-0.25, 0, 1.68], rgba=[0.1, 0.75, 0.95, 1], **kw)
            cam_pos = [0.6, 0, 1.55]
            b.add_camera(name="front", pos=cam_pos, xyaxes=lookat_xyaxes(cam_pos, [20.6, 0, 0.3]), fovy=50)
            if av_rig:                                   # Alpamayo's six-camera rig (pinhole approximations)
                for nm, pos, yaw_deg, hfov in AV_RIG:
                    yaw = np.radians(yaw_deg)
                    tgt = np.array(pos) + [20 * np.cos(yaw), 20 * np.sin(yaw), -0.6]
                    vfov = np.degrees(2 * np.arctan(np.tan(np.radians(hfov) / 2) * 9 / 16))
                    b.add_camera(name=nm, pos=pos, xyaxes=lookat_xyaxes(pos, tgt), fovy=float(vfov))
        return b
    car_body("ego", (0.93, 0.94, 0.96), ego=True)
    for k in range(n_cars):
        car_body(f"car{k}", CAR_COLORS[k % len(CAR_COLORS)])
    for k in range(n_peds + 1):                                     # the last one is the jaywalker
        b = wb.add_body(name=f"ped{k}", mocap=True, pos=[0, 0, -10])
        col = PED_COLORS[k % len(PED_COLORS)] if k < n_peds else (0.95, 0.35, 0.65)
        kw = dict(contype=0, conaffinity=0)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.2, 0.32], pos=[0, 0, 1.08], rgba=_hex(col), **kw)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.09, 0.36], pos=[0, 0.1, 0.42], rgba=[0.15, 0.15, 0.2, 1], name=f"legL{k}", **kw)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.09, 0.36], pos=[0, -0.1, 0.42], rgba=[0.15, 0.15, 0.2, 1], name=f"legR{k}", **kw)
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.14], pos=[0, 0, 1.62], rgba=[0.9, 0.75, 0.62, 1], **kw)
    model = spec.compile()
    lamp_ids = {k: model.geom(n).id for k, n in lamps.items()}
    return model, lamp_ids


LAMP_Z = (5.65, 5.0, 4.35)
# Alpamayo's camera rig: (name, position on the car, yaw in degrees (left +), horizontal field of view). The camera ids
# in the model's prompt: 0 cross-left, 1 front-wide, 2 cross-right, 3 rear-left, 5 rear-right, 6 front-tele.
AV_RIG = [("av0_cross_left", [1.9, 0.8, 1.2], 70, 120), ("av1_front_wide", [1.7, 0, 1.45], 0, 120),
          ("av2_cross_right", [1.9, -0.8, 1.2], -70, 120), ("av3_rear_left", [0.9, 1.0, 1.15], 145, 70),
          ("av5_rear_right", [0.9, -1.0, 1.15], -145, 70), ("av6_front_tele", [1.7, 0, 1.45], 0, 30)]
LAMP_RGBA = np.array([[1.0, 0.12, 0.08, 1], [1.0, 0.78, 0.1, 1], [0.15, 1.0, 0.35, 1]], np.float32)


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])


# ---------------------------------------------------------------- the driving environment
class DriveEnv:
    """One driving episode in the city. The ego car follows a route through two intersections.

    obs = dict(images={"front": HxWx3}, speed, instruction, command ("L"/"S"/"R"/"F"), t)
    step(action=(acceleration, curvature)) -> obs, done, info
    info: success, collision, offroad, offroute, red_light (violations so far), progress (m), reason (expert only)
    """

    def __init__(self, n_cars=24, n_peds=40, cam_size=(128, 224), render=True, city_seed=0, av_rig=False):
        self.n_cars, self.n_peds, self.cam_size = n_cars, n_peds, cam_size
        self.model, self.lamp_ids = build_city(n_cars, n_peds, seed=city_seed, av_rig=av_rig)
        self.data = mujoco.MjData(self.model)
        self.ego_mocap = self.model.body("ego").mocapid[0]
        self.car_mocap = [self.model.body(f"car{k}").mocapid[0] for k in range(n_cars)]
        self.ped_mocap = [self.model.body(f"ped{k}").mocapid[0] for k in range(n_peds + 1)]
        self.legs = [(self.model.geom(f"legL{k}").id, self.model.geom(f"legR{k}").id) for k in range(n_peds + 1)]
        self.renderer = mujoco.Renderer(self.model, *cam_size) if render else None
        self._substep_cb = None
        self.render_obs = True              # step() renders the policy camera (set False when recording sparsely)

    # ------------------------------------------------------------ episode setup
    def reset(self, seed=0, scenario=None):
        """scenario: None (random), or one of 'left', 'right', 'straight', 'jaywalker', 'red' to force a situation."""
        rng = self.rng = np.random.default_rng(seed)
        self.t, self.lights = 0.0, Lights(rng)
        # the ego route: a start road, two intersections, a final stretch
        for _ in range(200):
            node = (int(rng.integers(N)), int(rng.integers(N)))
            d = int(rng.integers(4))
            if not in_grid(step_node(node, d)):
                continue
            dirs, ok, nd, dd = [], True, node, d
            want = {"left": "L", "right": "R", "straight": "S"}.get(scenario)
            for k in range(2):
                nb = step_node(nd, dd)
                opts = [d2 for d2 in range(4) if d2 != (dd + 2) % 4 and in_grid(step_node(nb, d2))]
                if k == 0 and want:
                    opts = [d2 for d2 in opts if turn_type(dd, d2) == want]
                    if want == "L":                         # a left turn with an oncoming road
                        opts = [d2 for d2 in opts if in_grid(step_node(nb, dd))]
                if not opts:
                    ok = False; break
                w = np.array([1.0 if turn_type(dd, d2) == "S" else 1.0 for d2 in opts])
                d2 = opts[rng.choice(len(opts), p=w / w.sum())]
                dirs.append(d2); nd, dd = nb, d2
            if ok:
                break
        along0 = rng.uniform(RW + 4, PITCH - RW - 34) if scenario != "jaywalker" else rng.uniform(RW + 2, RW + 6)
        self.route = Path(node, d, along0, dirs, end_along=None)          # ends at the entry of a third intersection
        self.route_end = self.route.junctions[-1]["s_out"] + 25.0          # the goal: 25 m after the second one
        self.commands = [J["turn"] for J in self.route.junctions]
        # ego state
        p, yaw = self.route.pose(0.0)
        self.ego = dict(xy=p.copy(), yaw=float(yaw), v=float(rng.uniform(3.0, 8.0)), kappa=0.0, s=0.0)
        if scenario == "red":
            J = self.route.junctions[0]
            # time the lights so the first light turns red as the car arrives
            arrive = (J["s_stop"] - 0.0) / max(self.ego["v"], 6.0)
            axis = J["d_in"] % 2
            target_u = G_T + Y_T * 0.2 if axis == 0 else (G_T + Y_T * 0.2 + G_T + Y_T + AR_T) % CYCLE
            self.lights.off[J["node"]] = (target_u - arrive) % CYCLE
        # traffic cars
        self.cars, occupied = [], [(self.ego["xy"], self.route)]
        tries = 0
        while len(self.cars) < self.n_cars and tries < 2000:
            tries += 1
            nd = (int(rng.integers(N)), int(rng.integers(N)))
            dd = int(rng.integers(4))
            if not in_grid(step_node(nd, dd)):
                continue
            along = rng.uniform(RW + 3, PITCH - RW - 12)
            xy = lane_pt(nd, dd, along)
            if any(np.linalg.norm(xy - o[0]) < 12 for o in occupied):
                continue
            if np.linalg.norm(xy - self.ego["xy"]) < 30:
                continue
            path = Path(nd, dd, along, random_dirs(nd, dd, 16, rng, allow_left=False), None)
            self.cars.append(Car(path, 0.0, rng.uniform(0, 7), rng.uniform(7.0, 10.0), None))
            occupied.append((xy, path))
        self.peds = [Ped(rng) for _ in range(self.n_peds)]
        self.jay = None
        if scenario == "jaywalker" or (scenario is None and rng.random() < 0.3):
            J0 = self.route.junctions[0]
            s_c = rng.uniform(22, max(23, J0["s_stop"] - 10)) if J0["s_stop"] > 35 else J0["s_out"] + rng.uniform(18, 30)
            self.jay = Jaywalker(self.route, s_c, rng)
        self.events = dict(collision=False, offroad=False, offroute=False, red_light=0, red_seen=set())
        self.done, self.success, self.dist, self.steps, self.lat = False, False, 0.0, 0, 0.0
        self._sync()
        return self.observe()

    # ------------------------------------------------------------ helpers
    @property
    def command(self):
        J = self.route.next_junction(self.ego["s"], margin=3.0)
        return J["turn"] if J is not None else "F"

    @property
    def instruction(self):
        return COMMANDS[self.command]

    def _oncoming(self, J):
        """Is there oncoming traffic a left turn at junction J must wait for?"""
        opp = (J["d_in"] + 2) % 4
        for c in self.cars:
            Jc = c.path.next_junction(c.s, margin=0.0)
            if Jc is None or Jc["node"] != J["node"] or Jc["d_in"] != opp:
                continue
            if c.s > Jc["s_in"] - 0.5:                       # already in the junction
                return True
            if (Jc["s_in"] - c.s) / max(c.v, 0.5) < 5.0 and self.lights.state(J["node"], opp, self.t) != "R":
                return True
        return False

    def agents(self, exclude_car=None, include_ego=True):
        xy = np.array([c.xy for c in self.cars]).reshape(-1, 2)
        vel = np.array([c.vel for c in self.cars]).reshape(-1, 2)
        if include_ego:
            e = self.ego
            xy = np.vstack([xy, e["xy"]]); vel = np.vstack([vel, e["v"] * np.array([np.cos(e["yaw"]), np.sin(e["yaw"])])])
        if exclude_car is not None:
            keep = np.arange(len(xy)) != exclude_car
            xy, vel = xy[keep], vel[keep]
        return xy, vel

    def ped_xy(self):
        p = [q.pos for q in self.peds]
        if self.jay is not None:
            p.append(self.jay.pos)
        return np.array(p).reshape(-1, 2)

    def expert(self):
        """The privileged driver: (acceleration, curvature) and the reason, from the true state of the world."""
        e = self.ego
        xy, vel = self.agents(include_ego=False)
        a, reason = longitudinal(self.route, e["s"], e["v"], 9.0, self.t, self.lights, xy, vel, self.ped_xy(),
                                 oncoming_fn=self._oncoming)
        k = pure_pursuit(self.route, e["s"], e["xy"], e["yaw"], e["v"])
        return np.array([a, k]), reason

    # ------------------------------------------------------------ simulation
    def step(self, action):
        a = float(np.clip(action[0], A_MIN, A_MAX))
        k_cmd = float(np.clip(action[1], -K_MAX, K_MAX))
        # traffic decisions at 10 Hz
        peds = self.ped_xy()
        all_xy, all_vel = self.agents()
        acc = []
        for i, c in enumerate(self.cars):
            keep = np.arange(len(all_xy)) != i
            acc.append(longitudinal(c.path, c.s, c.v, c.v_des, self.t, self.lights, all_xy[keep], all_vel[keep], peds)[0])
        e = self.ego
        h = DT / SUB
        speeds = np.r_[[c.v for c in self.cars], e["v"]]
        for _ in range(SUB):
            self.t += h
            for c, ac in zip(self.cars, acc):
                c.v = max(0.0, c.v + ac * h); c.a = ac
                c.move_to(c.s + c.v * h)
            cxy = np.vstack([[c.xy for c in self.cars], e["xy"][None]])
            for p in self.peds:
                p.update(h, self.t, self.lights, cxy, speeds)
            if self.jay is not None:
                self.jay.update(h, e["s"], e["v"], cxy, speeds)
            dk = np.clip(k_cmd - e["kappa"], -K_RATE * h, K_RATE * h)
            e["kappa"] += dk
            e["v"] = max(0.0, e["v"] + a * h)
            e["yaw"] += e["v"] * e["kappa"] * h
            step = e["v"] * h
            e["xy"] = e["xy"] + step * np.array([np.cos(e["yaw"]), np.sin(e["yaw"])])
            self.dist += step
            e["s"], self.lat = self.route.project(e["xy"], e["s"])
            self._check()
            self._sync()
            if self._substep_cb is not None:
                self._substep_cb(self)
        self.steps += 1
        if e["s"] >= self.route_end:
            self.done, self.success = True, not (self.events["collision"] or self.events["offroad"])
        if self.events["collision"] or self.events["offroad"] or self.events["offroute"] or self.t > 90:
            self.done = True
        return self.observe(render=self.render_obs), self.done, self.info()

    def _check(self):
        e, ev = self.ego, self.events
        ce = car_circles(e["xy"], e["yaw"])
        cxy = np.array([c.xy for c in self.cars])
        near = np.linalg.norm(cxy - e["xy"], axis=1) < 6.0
        for k in np.where(near)[0]:
            c = self.cars[k]
            if np.min(np.linalg.norm(ce[:, None] - car_circles(c.xy, c.yaw)[None], axis=2)) < 1.85:
                ev["collision"] = "car"
        P = self.ped_xy()
        if len(P) and np.min(np.linalg.norm(ce[:, None] - P[None], axis=2)) < 1.2:
            ev["collision"] = "pedestrian"
        if not on_road(e["xy"]):
            ev["offroad"] = True
        if abs(self.lat) > 4.0:
            ev["offroute"] = True
        J = self.route.next_junction(e["s"])
        if J is not None and J["s_stop"] + 1.0 < e["s"] + CAR_L / 2 < J["s_stop"] + 2.5 and J["node"] not in ev["red_seen"]:
            if self.lights.state(J["node"], J["d_in"], self.t) == "R":
                ev["red_light"] += 1
                ev["red_seen"].add(J["node"])

    def info(self):
        return dict(success=self.success, collision=self.events["collision"], offroad=self.events["offroad"],
                    offroute=self.events["offroute"], red_light=self.events["red_light"], progress=self.ego["s"],
                    distance=self.dist, t=self.t)

    # ------------------------------------------------------------ visuals
    def _sync(self):
        d, e = self.data, self.ego
        d.mocap_pos[self.ego_mocap] = [e["xy"][0], e["xy"][1], 0.0]
        d.mocap_quat[self.ego_mocap] = _yaw_quat(e["yaw"])
        for k, c in enumerate(self.cars):
            d.mocap_pos[self.car_mocap[k]] = [c.xy[0], c.xy[1], 0.0]
            d.mocap_quat[self.car_mocap[k]] = _yaw_quat(c.yaw)
        walkers = list(self.peds) + ([self.jay] if self.jay is not None else [])
        for k in range(len(self.ped_mocap)):
            mid = self.ped_mocap[k]
            if k < len(walkers):
                w = walkers[k]
                yaw = np.arctan2(w.vel[1], w.vel[0]) if np.linalg.norm(w.vel) > 0.1 else 0.0
                d.mocap_pos[mid] = [w.pos[0], w.pos[1], 0.12 + 0.03 * abs(np.sin(w.phase))]
                d.mocap_quat[mid] = _yaw_quat(yaw)
                sw = 0.35 * np.sin(w.phase) if np.linalg.norm(w.vel) > 0.1 else 0.0
                for g, sgn in zip(self.legs[k], (1, -1)):
                    self.model.geom_pos[g][0] = sgn * sw * 0.4
            else:
                d.mocap_pos[mid] = [0, 0, -10]
        for (i, j, din), g in self.lamp_ids.items():
            on = {"R": 0, "Y": 1, "G": 2}[self.lights.state((i, j), din, self.t)]
            self.model.geom_pos[g][2] = LAMP_Z[on]
            self.model.geom_rgba[g] = LAMP_RGBA[on]
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_camlight(self.model, d)

    def render(self, camera="front", renderer=None):
        r = renderer or self.renderer
        r.update_scene(self.data, camera)
        return r.render().copy()

    def observe(self, render=True):
        img = self.render() if (render and self.renderer is not None) else None
        return dict(images={"front": img}, speed=self.ego["v"], instruction=self.instruction, command=self.command, t=self.t)


# ---------------------------------------------------------------- demonstrations
class OUNoise:
    def __init__(self, rng, sigma, theta=0.6):
        self.rng, self.sigma, self.theta, self.x = rng, np.asarray(sigma, float), theta, np.zeros(len(sigma))

    def __call__(self):
        self.x += -self.theta * self.x * DT + self.sigma * np.sqrt(DT) * self.rng.standard_normal(len(self.x))
        return self.x


def record_episode(env, seed, noise=True, img_every=2, scenario=None):
    """Drive one episode with the expert. With noise, the EXECUTED actions get correlated noise (DART), while the
    recorded LABELS stay the expert's clean actions, so the data shows how to recover from small mistakes.
    Returns dict of arrays (images every img_every control steps)."""
    obs = env.reset(seed=seed, scenario=scenario)
    rng = np.random.default_rng(seed + 7)
    ou = OUNoise(rng, sigma=[0.6, 0.03]) if noise else None
    rec = dict(img=[], img_step=[], speed=[], a=[], k=[], reason=[], cmd=[], ok=False)
    done, step = False, 0
    env.render_obs = False
    still = 0
    while not done:
        act, reason = env.expert()
        still = still + 1 if env.ego["v"] < 0.05 else 0
        # a camera frame every img_every steps; while waiting (stopped > 1 s) only every 3rd one: those frames repeat
        if step % img_every == 0 and (still < 10 or step % (3 * img_every) == 0):
            rec["img"].append(env.render()); rec["img_step"].append(step)
        rec["speed"].append(env.ego["v"]); rec["a"].append(act[0]); rec["k"].append(act[1])
        rec["reason"].append(REASONS.index(reason)); rec["cmd"].append("LSRF".index(obs["command"]))
        noise = ou() if ou is not None else 0
        exe = act + (noise if env.ego["v"] > 0.5 else 0)                  # no noise while standing still
        obs, done, info = env.step(exe)
        step += 1
    env.render_obs = True
    rec["ok"] = bool(info["success"])
    out = {k: np.asarray(v) for k, v in rec.items() if k != "ok"}
    out["img"] = np.stack(rec["img"]).astype(np.uint8)
    out["success"] = np.array(rec["ok"])
    return out, info


def _record_worker(args):
    out_dir, seeds, noise_frac, cam_size = args
    os.environ.setdefault("MUJOCO_GL", "egl")
    env = DriveEnv(cam_size=tuple(cam_size))
    kept = []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        ep, info = record_episode(env, sd, noise=rng.random() < noise_frac)
        if not ep["success"]:
            continue
        np.savez_compressed(os.path.join(out_dir, f"ep_{sd:06d}.npz"), **ep)
        kept.append(dict(seed=int(sd), frames=int(len(ep["img"])), steps=int(len(ep["a"]))))
    return kept


def collect_drives(out_dir, n_episodes, workers=8, seed0=0, noise_frac=0.6, cam_size=(128, 224)):
    """Record expert episodes in parallel (spawned workers: safe with GPU rendering). Failed episodes are discarded."""
    import multiprocessing as mp
    os.makedirs(out_dir, exist_ok=True)
    seeds = list(range(seed0, seed0 + n_episodes))
    chunks = [(out_dir, seeds[w::workers], noise_frac, list(cam_size)) for w in range(workers)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers) as pool:
        kept = [e for part in pool.map(_record_worker, chunks) for e in part]
    kept.sort(key=lambda e: e["seed"])
    meta = dict(n_episodes=len(kept), attempts=n_episodes, n_frames=sum(e["frames"] for e in kept),
                cam_size=list(cam_size), dt=DT, episodes=kept, seconds=time.time() - t0)
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(f"{len(kept)}/{n_episodes} expert episodes kept, {meta['n_frames']} frames, {meta['seconds']:.0f}s")
    return meta
