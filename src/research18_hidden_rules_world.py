"""A small 2D causal world; hidden physical state is evaluator-only."""
import numpy as np

RULES=("constant","drag","accelerate")

def episode(seed=1,steps=20,size=32,rule="constant",occlude=True,
            camera_actions=None):
    if rule not in RULES or steps<6 or size<16:raise ValueError("Invalid world")
    rng=np.random.default_rng(seed)
    p=np.array([size*.18,rng.uniform(size*.3,size*.7)])
    v=np.array([1.2,.2],dtype=float)
    actions=np.zeros((steps,2)) if camera_actions is None else np.asarray(camera_actions,float)
    if actions.shape!=(steps,2):raise ValueError("Expected known per-step camera displacement")
    camera=np.zeros(2)
    images=[];truth=[];visible=[];frames=[]
    yy,xx=np.mgrid[:size,:size]
    switch=steps//3
    for t in range(steps):
        if t==switch:
            if rule=="drag":v*=.6
            if rule=="accelerate":v*=1.6
        camera+=actions[t]
        screen=p-camera
        pixels=((xx-screen[0])**2+(yy-screen[1])**2<=2.5**2)
        covered=np.logical_and(pixels,np.abs(xx-size//2)<size*.12) if occlude else np.zeros_like(pixels)
        image=pixels.astype(float)
        if occlude:image[:,(np.abs(np.arange(size)-size//2)<size*.12)]=.16
        if occlude:image[covered]=.16
        images.append(image)
        truth.append(p.copy())
        visible.append(bool(np.any(pixels&~covered)))
        frames.append(camera.copy())
        p=p+v
        for a in range(2):
            if p[a]<3 or p[a]>size-4:
                v[a]*=-1
                p[a]=np.clip(p[a],3,size-4)
    return {"observations":np.stack(images),
            "model_actions":actions.copy(),
            "evaluator_only":{"positions":np.stack(truth),
                              "visible":np.asarray(visible),
                              "camera":np.stack(frames),
                              "rule":rule,"switch_time":switch},
            "seed":seed}

def paired_rules(seed=1,steps=20):
    a=episode(seed,steps,rule="constant")
    b=episode(seed,steps,rule="accelerate")
    return a,b

def prediction_mse(predicted,target):
    a=np.asarray(predicted,float);b=np.asarray(target,float)
    if a.shape!=b.shape:raise ValueError("Shape mismatch")
    return float(np.mean((a-b)**2))
