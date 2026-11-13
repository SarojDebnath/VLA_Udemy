# %% [markdown]
# # 117 · LeRobot (kernel: Python (lerobot)): export TableTop to the official LeRobotDataset v3.0 and load it back

# %%
import os, glob, json, shutil, time, numpy as np, torch, matplotlib.pyplot as plt
from lerobot.datasets.lerobot_dataset import LeRobotDataset, CODEBASE_VERSION

SRC = os.path.abspath("../data/tabletop_panda_v1")
ROOT = os.path.abspath("../data/lerobot_v3/tabletop_panda_v1")
FEATURES = {
    "observation.images.front": {"dtype": "video", "shape": (128, 128, 3), "names": ["height", "width", "channels"]},
    "observation.images.wrist": {"dtype": "video", "shape": (128, 128, 3), "names": ["height", "width", "channels"]},
    "observation.state": {"dtype": "float32", "shape": (8,), "names": ["ee_x", "ee_y", "ee_z", "qw", "qx", "qy", "qz", "gripper_width"]},
    "action": {"dtype": "float32", "shape": (7,), "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"]},
}

def export(src, root):
    shutil.rmtree(root, ignore_errors=True)
    ds = LeRobotDataset.create(repo_id="local/tabletop_panda_v1", fps=10, features=FEATURES, root=root, robot_type="franka_panda", use_videos=True, vcodec="h264")
    for path in sorted(glob.glob(f"{src}/episode_*.npz")):
        z = np.load(path)
        for t in range(len(z["action"])):
            ds.add_frame({"observation.images.front": z["front"][t], "observation.images.wrist": z["wrist"][t],
                          "observation.state": z["state"][t], "action": z["action"][t], "task": str(z["instruction"])})
        ds.save_episode()
    ds.finalize()
    return ds

# %%
print("LeRobot dataset codebase version:", CODEBASE_VERSION)
if not os.path.exists(f"{ROOT}/meta/info.json"):
    t0 = time.time(); export(SRC, ROOT); print(f"exported in {time.time() - t0:.0f}s")
info = json.load(open(f"{ROOT}/meta/info.json"))
print({k: info[k] for k in ["codebase_version", "robot_type", "total_episodes", "total_frames", "total_tasks", "fps"]})
for r, _, fs in os.walk(ROOT):
    if fs:
        print(f"  {r[len(ROOT):] or '/'}: {fs[:3]}{' ...' if len(fs) > 3 else ''}")

# %%
H = 16
# video_backend="pyav": PyAV bundles its own FFmpeg. The default (torchcodec) needs system FFmpeg 4-8 shared libraries,
# which many machines (and Colab images) don't have: "Could not load libtorchcodec ... libavutil.so" means exactly that.
ds = LeRobotDataset("local/tabletop_panda_v1", root=ROOT, delta_timestamps={"action": [i / 10 for i in range(H)], "observation.images.front": [-0.1, 0.0]},
                    video_backend="pyav")
s = ds[123]
print({k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in s.items()})
print("action_is_pad:", s["action_is_pad"].int().tolist())
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(s["observation.images.front"][0].permute(1, 2, 0)); ax[0].set_title("front, t-1 (delta -0.1 s)")
ax[1].imshow(s["observation.images.front"][1].permute(1, 2, 0)); ax[1].set_title("front, t")
ax[2].plot(s["action"][:, :3]); ax[2].set_title("action chunk a[t:t+16] (dx, dy, dz)")
plt.suptitle(s["task"]); plt.tight_layout(); plt.show()

# %%
stats = ds.meta.stats
print("action mean:", np.round(np.array(stats["action"]["mean"]), 4))
print("action std :", np.round(np.array(stats["action"]["std"]), 4))
print("tasks:", list(ds.meta.tasks.index)[:5], "...")
dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
b = next(iter(dl))
print("batch:", {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in b.items()})
print("To share: ds.push_to_hub()  (requires `huggingface-cli login`); training a LeRobot policy: lerobot-train --dataset.repo_id=local/tabletop_panda_v1 --dataset.root=" + ROOT)
