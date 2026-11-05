"""Matched impulse experiments for physical field spread and recovery.

Positive controls are checked against the *numerical world*, not learned
predictions. Compare a local physical impulse against an equal-L2 displaced
impulse and a zero-perturbation control.
"""
import numpy as np
from src import physical_fields as physics


def pulse(size=24,amplitude=.5,position=None):
    if size<8 or amplitude<=0:
        raise ValueError("Invalid pulse")
    y,x=(size//2,size//2) if position is None else tuple(position)
    if not 0<=y<size or not 0<=x<size:
        raise ValueError("Position outside world")
    field=np.zeros((size,size),np.float64)
    field[y,x]=amplitude
    return field


def propagate(seed=11,kind="diffusion",size=24,steps=24,
              location=None,amplitude=.5,component="displacement",
              diffusivity=.15,speed=.5,damping=.02):
    if steps<2 or component not in ("displacement","velocity"):
        raise ValueError("Invalid impulse protocol")
    if kind=="diffusion" and component=="velocity":
        raise ValueError("Heat equation has no velocity component")
    source=physics.episode(seed,kind,size,steps,diffusivity=diffusivity,
                           speed=speed,damping=damping)
    baseline=source["states"][0]
    changed=baseline.copy()
    index=1 if component=="velocity" else 0
    changed[index]+=pulse(size,amplitude,location)
    interventions=[]
    states=[]
    for starting,collection in ((baseline,states),(changed,interventions)):
        x=starting
        collection.append(x.copy())
        for _ in range(steps-1):
            x=physics.step(x,kind=kind,diffusivity=diffusivity,
                           speed=speed,damping=damping)
            collection.append(x.copy())
    normal=np.stack(states)
    changed=np.stack(interventions)
    response=changed-normal
    # Diagnostics operate on the physical impulse response, not the
    # possibly large unperturbed moving fields.
    yy,xx=np.mgrid[:size,:size]
    center=(size//2,size//2) if location is None else location
    distance=np.sqrt((yy-center[0])**2+(xx-center[1])**2)
    squared=response[:,0]**2
    energy=squared.sum(axis=(1,2))
    spread=(squared*distance[None]**2).sum(axis=(1,2))/np.maximum(energy,1e-12)
    return {"baseline":normal,"intervened":changed,
            "response":response,
            "mean_squared_radius":spread,
            "response_l2":np.sqrt(np.sum(response**2,axis=(1,2,3))),
            "response_max_abs":np.max(np.abs(response),axis=(1,2,3)),
            "parameters":{"kind":kind,"seed":int(seed),"steps":steps,
                          "size":size,"location":list(center),
                          "amplitude":amplitude,"component":component,
                          "diffusivity":diffusivity,"speed":speed,
                          "damping":damping}}


def calibrate(seed=11,size=24,steps=20):
    heat=propagate(seed,"diffusion",size,steps)
    wave=propagate(seed,"wave",size,steps,component="velocity")
    zero=heat["baseline"]-heat["baseline"]
    # Center pulse impulse in diffusion should conserve its integrated
    # perturbation under reflecting boundaries and spread spatially.
    heat_mass=heat["response"][:,0].sum(axis=(1,2))
    if not np.allclose(heat_mass,heat_mass[0],atol=1e-9):
        raise AssertionError("Diffusion perturbation mass not conserved")
    if heat["mean_squared_radius"][-1]<=heat["mean_squared_radius"][0]:
        raise AssertionError("Diffusion perturbation failed to spread")
    if not np.max(np.abs(wave["response"][:,0]))>0:
        raise AssertionError("Velocity impulse did not reach wave displacement")
    return {"heat_impulse_mass_max_error":float(np.max(abs(heat_mass-heat_mass[0]))),
            "diffusion_start_radius2":float(heat["mean_squared_radius"][0]),
            "diffusion_final_radius2":float(heat["mean_squared_radius"][-1]),
            "wave_displacement_response_max":float(np.max(abs(wave["response"][:,0]))),
            "zero_control_l2":float(np.linalg.norm(zero))}


def match_controls(seed=11,kind="diffusion",size=24,steps=20):
    center=propagate(seed,kind,size,steps)
    shifted=propagate(seed,kind,size,steps,
                      location=(size//2,size//2+3))
    zero=np.zeros_like(center["response"])
    if not np.isclose(np.linalg.norm(center["response"][0]),
                      np.linalg.norm(shifted["response"][0])):
        raise AssertionError("Intervention energies mismatched")
    return {"center":center,"shifted":shifted,"zero":zero}
