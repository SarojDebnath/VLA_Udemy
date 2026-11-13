# %% [markdown]
# # 27 · Interaction: passive viewer, key callbacks, mocap bodies, recording video

# %%
import os, time, mujoco, numpy as np
from vlacourse import TableTop, show_video, save_video
from vlacourse.panda import solve_ik

HEADLESS = os.environ.get("VLA_HEADLESS") == "1"

def mocap_scene():
    xml = """<mujoco><option timestep="0.002"/>
    <worldbody><light pos="0 0 2"/><geom type="plane" size="1 1 .1" rgba=".3 .35 .4 1"/>
      <body name="target" mocap="true" pos="0 0 .5"><geom type="sphere" size=".03" rgba="0 1 0 .5" contype="0" conaffinity="0"/></body>
      <body name="ball" pos="0 0 .5"><freejoint/><geom type="sphere" size=".05" rgba="1 .3 .3 1" mass=".2"/></body>
    </worldbody>
    <equality><weld body1="target" body2="ball" solref=".05 1"/></equality></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    return m, mujoco.MjData(m)

# %%
m, d = mocap_scene()
frames = []
with mujoco.Renderer(m, 300, 400) as r:
    while d.time < 4:
        d.mocap_pos[0] = [0.3 * np.cos(2 * d.time), 0.3 * np.sin(2 * d.time), 0.4 + 0.1 * np.sin(4 * d.time)]
        mujoco.mj_step(m, d)
        if len(frames) < d.time * 30:
            cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, 0.3]; cam.distance, cam.elevation = 1.8, -30
            r.update_scene(d, cam); frames.append(r.render())
print("mocap target drags the ball through a soft weld constraint; frames:", len(frames))
show_video(frames, fps=30, title="mocap follow")

# %%
env = TableTop(seed=0, cameras=("front", "wrist"), image_size=(240, 320))
frames = []
for k in range(40):
    action = np.r_[0.01 * np.sin(k / 4), 0.01 * np.cos(k / 4), -0.003, 0, 0, 0.05, 0.0]
    obs, *_ = env.step(action)
    frames.append(np.hstack([obs["images"]["front"], obs["images"]["wrist"]]))
path = save_video(frames, "generated/tabletop_wiggle.mp4", fps=10)
print("saved", path, os.path.getsize(path) // 1024, "KB")
show_video(frames, fps=10, title="front | wrist")

# %%
keys = {"W": (0.01, 0, 0), "S": (-0.01, 0, 0), "A": (0, 0.01, 0), "D": (0, -0.01, 0), "Q": (0, 0, 0.01), "E": (0, 0, -0.01)}
pending = []

def on_key(keycode):
    ch = chr(keycode) if 32 <= keycode < 127 else ""
    if ch in keys:
        pending.append(keys[ch])
    elif ch == " ":
        env.grip = 1.0 - env.grip

if not HEADLESS:
    import mujoco.viewer
    env.reset(seed=1)
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
        t_end = time.time() + 60
        while viewer.is_running() and time.time() < t_end:
            dpos = np.sum(pending, axis=0) if pending else np.zeros(3)
            pending.clear()
            with viewer.lock():
                env.step(np.r_[dpos, 0, 0, 0, env.grip])
            viewer.sync()
            time.sleep(0.05)
else:
    print("headless: skipping the interactive viewer (W/S/A/D/Q/E move, SPACE toggles the gripper)")
