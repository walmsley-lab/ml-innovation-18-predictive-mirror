"""Two-object occlusion and an explicitly unidentifiable identity swap.

Generator truth is evaluator-only. Distinct colors permit appearance-based
identity; identical-color hidden swaps do not provide image evidence.
"""
import numpy as np

COLORS=np.array([[.94,.18,.14],[.15,.78,.95]],np.float32)


def episode(seed=11,context=8,horizon=10,size=32,appearance="distinct",
            hidden_swap=False):
    if (size<32 or context<8 or horizon<6
            or appearance not in ("distinct","identical")):
        raise ValueError("Invalid identity world")
    tlen=context+horizon
    rng=np.random.default_rng(int(seed))
    cy=float(size//2+rng.integers(-1,2))
    mid=(size-1)/2.
    speed=2.
    frames,positions,visible=[],[],[]
    for t in range(tlen):
        x0=mid-speed*(context-1-t)
        x1=mid+speed*(context-1-t)
        if hidden_swap and t>=context:
            # A hidden identity exchange: physical positions, and therefore
            # rendered pixels with identical colors, are unchanged.
            xy=np.array([[x1,cy],[x0,cy]],float)
        else:
            xy=np.array([[x0,cy],[x1,cy]],float)
        frame=np.full((size,size,3),.035,np.float32)
        yy,xx=np.mgrid[:size,:size]
        mask=np.abs(xx-mid)<=5.0
        present=[]
        for i,(x,y) in enumerate(xy):
            disc=(xx-x)**2+(yy-y)**2<=2.2**2
            rgb=COLORS[i] if appearance=="distinct" else COLORS[0]
            frame[disc]=rgb
            present.append(bool(np.any(disc&~mask)))
        frame[mask]=[.12,.12,.15]
        frames.append(frame)
        positions.append(xy)
        visible.append(present)
    return {"frames":np.stack(frames),"positions":np.stack(positions),
            "visible":np.asarray(visible),
            "seed":int(seed),"context":context,"horizon":horizon,
            "appearance":appearance,"hidden_swap":bool(hidden_swap),
            "identity_observable_from_rgb":appearance=="distinct"}


def paired_hidden_identity(seed=11):
    """Two identity assignments with exactly identical RGB observations."""
    first=episode(seed,appearance="identical",hidden_swap=False)
    other=episode(seed,appearance="identical",hidden_swap=True)
    np.testing.assert_array_equal(first["frames"],other["frames"])
    if np.array_equal(first["positions"][first["context"]:],
                      other["positions"][other["context"]:]):
        raise AssertionError("Hidden identity alternatives must differ")
    return first,other


def identity_score(predicted_color_label,record):
    """Appearance-label accuracy is undefined for indistinguishable identities."""
    if not record["identity_observable_from_rgb"]:
        return None
    if predicted_color_label not in (0,1):
        raise ValueError("Expected label 0 or 1")
    return int(predicted_color_label==0)


def branch_brier(probability_swap,swapped):
    """Brier score for an explicitly defined binary hidden identity branch."""
    p=float(probability_swap)
    if not np.isfinite(p) or p<0 or p>1:
        raise ValueError("Need a probability in [0,1]")
    return float((p-int(bool(swapped)))**2)


def branch_log_loss(probability_swap,swapped,eps=1e-9):
    p=float(probability_swap)
    if not np.isfinite(p) or p<0 or p>1:
        raise ValueError("Need a probability in [0,1]")
    return -float(np.log(np.clip(p if swapped else 1-p,eps,1)))
