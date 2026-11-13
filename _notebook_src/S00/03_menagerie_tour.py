# %% [markdown]
# # 03 · MuJoCo Menagerie tour: the robots of this course

# %%
import mujoco, mujoco_menagerie as mm, numpy as np, matplotlib.pyplot as plt

ROBOTS = ["franka_emika_panda", "universal_robots_ur5e", "robotstudio_so101", "aloha",
          "unitree_go2", "unitree_g1", "shadow_hand", "leap_hand", "robotiq_2f85", "kinova_gen3"]

def snapshot(model, data, size=(320, 320), azimuth=135, elevation=-20, scale=1.6):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = model.stat.center
    cam.distance, cam.azimuth, cam.elevation = scale * model.stat.extent, azimuth, elevation
    with mujoco.Renderer(model, *size) as r:
        r.update_scene(data, cam)
        return r.render()

def load(name):
    model = mm.load(name)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data

# %%
for name in ROBOTS:
    r = mm.get(name)
    print(f"{r.display_name:28s} {r.category:10s} {r.license:15s} entries={r.entry_names}")

# %%
fig, axes = plt.subplots(2, 5, figsize=(18, 7.5))
for ax, name in zip(axes.flat, ROBOTS):
    model, data = load(name)
    ax.imshow(snapshot(model, data)); ax.axis("off")
    ax.set_title(f"{name}\nnq={model.nq} nv={model.nv} nu={model.nu}", fontsize=9)
plt.tight_layout(); plt.show()

# %%
model, data = load("franka_emika_panda")
for j in range(model.njnt):
    print(f"{model.joint(j).name:15s} type={model.jnt_type[j]} range={model.jnt_range[j].round(3)}")
for a in range(model.nu):
    print(f"{model.actuator(a).name:12s} ctrlrange={model.actuator_ctrlrange[a].round(3)}")
