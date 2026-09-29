"""Pick an offscreen OpenGL backend before mujoco is imported.

macOS uses CGL by default and needs nothing. Headless Linux (Colab) needs EGL on a GPU
runtime or OSMesa on a CPU runtime. An explicit MUJOCO_GL always wins.
"""

import os
import shutil
import sys

if sys.platform.startswith("linux") and "MUJOCO_GL" not in os.environ and not os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "egl" if shutil.which("nvidia-smi") else "osmesa"
    os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
