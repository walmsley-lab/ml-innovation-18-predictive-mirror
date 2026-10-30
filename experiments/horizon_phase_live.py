"""Experimental multi-horizon camera display; preserves the original mirror.

Runs the separately trained multi-horizon phase or plain recurrent checkpoint.
Future panes are model forecasts, never copied from future camera observations.
"""
import argparse
import time
from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

from experiments.horizon_phase import ARMS, rollout_one
from mirror import preprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", default="runs/horizon-phase/phase_coupled.npz")
    parser.add_argument("--arm", choices=ARMS, default="phase_coupled")
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--coupling", type=float, default=0.25)
    parser.add_argument("--panel-size", type=int, default=192)
    args = parser.parse_args()
    if args.context < 2 or args.horizon < 1 or args.fps <= 0 or args.size < 8:
        parser.error("Invalid context, horizon, FPS or image size")

    try:
        import cv2
    except ImportError:
        parser.error("Install opencv-python for local live display")
    with np.load(args.parameters, allow_pickle=False) as f:
        params = {key: jnp.asarray(f[key]) for key in f.files}
    predict = jax.jit(lambda x: rollout_one(
        params, x, args.horizon, args.arm, args.coupling))
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera {args.camera}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # advisory; driver may ignore
    observed = deque(maxlen=args.context)
    leads = tuple(sorted(set((1, 3, 5, 10, args.horizon)) & set(range(1, args.horizon+1))))

    def panel(frame, label):
        scaled = cv2.resize((255 * np.clip(frame[..., 0], 0, 1)).astype(np.uint8),
                            (args.panel_size, args.panel_size),
                            interpolation=cv2.INTER_NEAREST)
        out = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
        cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (75, 235, 228), 1)
        return out

    print("Experimental predictions (not verified real-world timings). Press q to quit.")
    deadline = time.monotonic()
    try:
        while True:
            wait = deadline - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            ok, raw = cap.read()
            if not ok:
                break
            observed.append(preprocess(raw, args.size, cv2))
            deadline = max(deadline + 1 / args.fps, time.monotonic())
            if len(observed) < args.context:
                continue
            stack = jnp.asarray(np.stack(observed))
            future = np.asarray(predict(stack))  # forecasts committed before next read
            panes = [panel(observed[-1], "Observed"),
                     panel(observed[-1], "Persistence")]
            panes += [panel(future[h-1], f"+{h} frames (~{h/args.fps:.1f}s)")
                      for h in leads]
            cv2.imshow("Long-Horizon Predictive Mirror (q to quit)",
                       np.hstack(panes))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
