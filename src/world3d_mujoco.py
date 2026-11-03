"""Optional MuJoCo RGB smoke backend; no dependency on the CPU test path.

MuJoCo v3+/4 Renderer APIs are version-dependent: pin and verify the
environment before treating an asset as a frozen benchmark.
"""
import numpy as np


XML = """<mujoco model="predictive-mirror-three-spheres">
  <visual><global offwidth="128" offheight="128"/></visual>
  <worldbody>
    <light pos="0 1 1" dir="0 0 -1"/>
    <camera name="eye" pos="0 0 0" quat="0 1 0 0" fovy="60"/>
    <geom type="sphere" pos="-0.7 0 3" size=".4" rgba=".95 .2 .12 1"/>
    <geom type="sphere" pos="0.4 0 4.5" size=".6" rgba=".12 .8 .95 1"/>
    <geom type="sphere" pos="0 0 6" size=".65" rgba=".9 .8 .1 1"/>
  </worldbody>
</mujoco>"""


def smoke(size=32, steps=4):
    """Render an actual MuJoCo 3D scene with a prescribed camera translation."""
    import mujoco  # Optional, deliberately not a CI requirement.
    if size < 12 or steps < 2:
        raise ValueError("Invalid MuJoCo smoke settings")
    model=mujoco.MjModel.from_xml_string(XML)
    data=mujoco.MjData(model)
    frames=[]
    renderer=mujoco.Renderer(model,height=size,width=size)
    try:
        for i in range(steps):
            model.cam_pos[0,0]=.10*i
            mujoco.mj_forward(model,data)
            renderer.update_scene(data,camera="eye")
            frames.append(renderer.render().copy().astype(np.float32)/255.)
    finally:
        renderer.close()
    return np.stack(frames)
