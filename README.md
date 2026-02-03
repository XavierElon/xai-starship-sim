# spaceX


## Create a training routine 
Routine that increases the initial rocket starting hight.
Starting at 5 once successfully trained go to 10 with the trained policy weights from 5 etc. 
Training should be faster and with the initialized weights and we can scale over time. 



# Setup

```bash
uv sync
source .venv/bin/activate
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