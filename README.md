# spaceX


# Setup

``` bash
conda create -n spaceX python=3.9
```
``` bash
conda activate spaceX
```

# GL MuJoCo Error

``` bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

# Use EGL for Headless Rendering
MuJoCo supports rendering through EGL, which is often compatible with headless servers. First, you need to ensure that MuJoCo is built with EGL support. Then, you can tell MuJoCo to use EGL instead of OpenGL by setting the MUJOCO_GL environment variable:

``` bash
export MUJOCO_GL=egl
```