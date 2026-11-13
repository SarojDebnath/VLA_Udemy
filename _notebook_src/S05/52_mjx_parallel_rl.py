# %% [markdown]
# # 52 · Massively parallel simulation with MJX: thousands of Unitree Go1s on one GPU (run on Colab GPU)

# %%
!pip install -q "playground==0.2.0" "jax[cuda12]==0.7.2" "flax==0.11.2" "orbax-checkpoint==0.11.22" mediapy
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

# %%
import time, functools, jax, jax.numpy as jp, numpy as np, mujoco, mediapy as media, matplotlib.pyplot as plt
from mujoco import mjx
from mujoco_playground import registry
print("devices:", jax.devices())

def bench(step_fn, state, n=100):
    state = step_fn(state); jax.block_until_ready(state)
    t0 = time.perf_counter()
    for _ in range(n):
        state = step_fn(state)
    jax.block_until_ready(state)
    return (time.perf_counter() - t0) / n

# %%
env = registry.load("Go1JoystickFlatTerrain")
mj_model = env.mj_model
mx = mjx.put_model(mj_model)
print(f"Go1: nq={mj_model.nq} nv={mj_model.nv} nu={mj_model.nu}  sim dt={env.sim_dt}  control dt={env.dt}")

# %%
cpu_data = mujoco.MjData(mj_model)
t0 = time.perf_counter(); mujoco.mj_step(mj_model, cpu_data, 2000); cpu_sps = 2000 / (time.perf_counter() - t0)
rows = [("CPU MuJoCo, 1 env", 1, cpu_sps)]
for n_envs in [1, 128, 1024, 4096]:
    keys = jax.random.split(jax.random.PRNGKey(0), n_envs)
    batch = jax.vmap(lambda k: mjx.make_data(mx).replace(qpos=jp.array(mj_model.qpos0) + 0.01 * jax.random.normal(k, (mj_model.nq,))))(keys)
    step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
    dt = bench(lambda d: step(mx, d), batch, 20)
    rows.append((f"MJX, {n_envs} envs", n_envs, n_envs / dt))
for name, n, sps in rows:
    print(f"{name:22s} {sps:14,.0f} steps/s")
plt.figure(figsize=(7, 4)); plt.bar([r[0] for r in rows], [r[2] for r in rows]); plt.yscale("log"); plt.ylabel("simulation steps / s"); plt.xticks(rotation=20); plt.show()

# %%
reset, stepf = jax.jit(env.reset), jax.jit(env.step)
state = reset(jax.random.PRNGKey(0)); traj = [state]
for i in range(150):
    state = stepf(state, jp.zeros(env.action_size)); traj.append(state)
media.show_video(env.render(traj[::2], height=360, width=640, camera="track"), fps=1 / env.dt / 2)
print("zero action = hold the default posture with the PD joint controllers")
