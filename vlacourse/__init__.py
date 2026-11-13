"""Shared helpers for the VLA course: TableTop world, Panda kinematics, scripted expert, video tools."""
import os as _os

# Headless Linux: make EGL use the NVIDIA driver. Without this, a user who isn't in the 'video'/'render' groups silently
# gets Mesa's CPU renderer (llvmpipe): ~500x slower rendering and the GPU sits at 0%. Must run before any GL context.
_NV = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
if _os.name != "nt" and _os.path.exists(_NV):
    _os.environ.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", _NV)
from .viz import show_video, save_video, show_images
from .panda import DOWN, solve_ik, ee_pose, add_ee_site, torque_panda, yaw_quat
from .tabletop import TableTop, COLORS, SCENES
from .expert import ScriptedExpert, ReactiveExpert
from .data import collect, collect_parallel, scene_dataset, list_episodes, load_episode, load_meta
from .evaluation import evaluate, wilson, ExpertPolicy, need
