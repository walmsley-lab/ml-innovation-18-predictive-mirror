"""Scan spectral modes, mode combinations, and phase-space helices on a graph.

This is an analysis of physical analog dynamics. It has no neural predictor.
For learning with a shared local cellular rule, run geometry_automata with
--medium harmonic to train on the same kind of broadband-driven trajectories.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src import geometry_automata as graph
from src import geometry_harmonics as harmonic


def root_mean_square(values):
    return float(np.sqrt(np.mean(np.square(values))))


def study(seed, nodes=16, kind="irregular", steps=720, dt=.055, out=None):
    if steps < 200 or nodes < 6 or dt <= 0:
        raise ValueError("Not enough samples for a spectral diagnostic")
    coords, edges = graph.generate(int(seed), nodes, kind)
    omega, basis = harmonic.modes(coords, edges)
    k, j = 1, 2
    point = int(np.argmax(abs(basis[:, k])))
    drive_site = np.eye(nodes)[point]
    resonant = harmonic.tone(steps, dt, omega[k], drive_site, .42)
    detuned = harmonic.tone(steps, dt, 1.6 * omega[k], drive_site, .42)
    q_on, v_on = harmonic.simulate(coords, edges, resonant, dt=dt)
    q_off, _ = harmonic.simulate(coords, edges, detuned, dt=dt)
    late = steps // 2
    gain = root_mean_square(q_on[late:] @ basis[:, k])
    offset_gain = root_mean_square(q_off[late:] @ basis[:, k])

    second_site = int(np.argmax(abs(basis[:, j])))
    secondary = harmonic.tone(steps, dt, omega[j],
                              np.eye(nodes)[second_site], .42)
    first, _ = harmonic.simulate(coords, edges, resonant, dt=dt)
    second, _ = harmonic.simulate(coords, edges, secondary, dt=dt)
    joint, _ = harmonic.simulate(coords, edges, resonant + secondary, dt=dt)
    # Exact linear superposition (to floating-point tolerance).
    superposition_error = root_mean_square(joint - first - second)
    first_n, _ = harmonic.simulate(coords, edges, resonant, dt=dt, cubic=2.)
    second_n, _ = harmonic.simulate(coords, edges, secondary, dt=dt, cubic=2.)
    joint_n, _ = harmonic.simulate(coords, edges, resonant + secondary,
                                   dt=dt, cubic=2.)
    nonlinear_difference = root_mean_square(joint_n - first_n - second_n)
    nonlinear_transfer = np.maximum(0., harmonic.modal_power(
        joint_n, basis, late) - harmonic.modal_power(joint, basis, late))

    # A drive-free/damped medium loses energy. External broadband noise is
    # an experimental forcing source, not an intrinsic universal substrate.
    noise_force = harmonic.broadband(int(seed)+7700, steps, nodes, .50)
    noise, _ = harmonic.simulate(coords, edges, noise_force, dt=dt, damping=.10)
    coefficient = noise @ basis[:, k]
    power = np.abs(np.fft.rfft(coefficient[late:] -
                               coefficient[late:].mean())) ** 2
    frequencies = np.fft.rfftfreq(len(coefficient[late:]), dt) * 2*np.pi
    peak = float(frequencies[1+np.argmax(power[1:])])

    helix, theta = harmonic.phase_trajectory(q_on, v_on, omega[k], basis[:, k])
    turns = float(abs(theta[-1]-theta[late])/(2*np.pi))

    shock_index = steps//2
    shock = np.zeros(nodes)
    shock[point] = .35
    changed, _ = harmonic.simulate(coords, edges, resonant, dt=dt,
                                   shock=(shock_index, shock))
    deviation = np.sqrt(np.mean((changed-q_on)**2, axis=1))
    initial_deviation = float(deviation[shock_index+1:shock_index+40].mean())
    end_deviation = float(deviation[-40:].mean())
    unforced, _ = harmonic.simulate(
        coords, edges, np.zeros_like(resonant), dt=dt,
        shock=(shock_index, shock))

    if out is not None:
        np.savez_compressed(
            out, coords=coords, edges=edges, eigenfrequencies=omega,
            basis=basis, on_resonance=q_on, detuned=q_off,
            broadband_response=noise, linear_two_tone=joint,
            nonlinear_two_tone=joint_n, modal_helix=helix,
            recovery_deviation=deviation, unforced_response=unforced)

    return {
        "seed": int(seed), "nodes": nodes, "geometry": kind,
        "target_mode": k, "target_frequency_rad_per_s": float(omega[k]),
        "detuned_frequency_rad_per_s": float(1.6*omega[k]),
        "resonant_modal_rms": gain, "detuned_modal_rms": offset_gain,
        "resonance_gain_ratio": gain/max(offset_gain, 1e-12),
        "linear_superposition_rmse": superposition_error,
        "nonlinear_superposition_rmse": nonlinear_difference,
        "nonlinear_extra_modal_power_sum": float(nonlinear_transfer.sum()),
        "broadband_modal_peak_rad_per_s": peak,
        "broadband_peak_error_rad_per_s": abs(peak-float(omega[k])),
        "phase_plane_winding_turns": turns,
        "shock_early_deviation": initial_deviation,
        "shock_late_deviation": end_deviation,
        "recovery_ratio": end_deviation/max(initial_deviation, 1e-12),
        "interpretation": "Linear modes are structurally predicted. Mode "
                          "response/recovery is externally forced and damped. "
                          "Nonlinear superposition failure is not proof of "
                          "emergent neural coordination. The helical chart "
                          "is constructed from measured modal phase."
    }


def run(out="runs/geometry-harmonics", seeds=(11, 23, 41),
        nodes=16, steps=720, dt=.055):
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Require distinct seeds")
    folder = Path(out)
    folder.mkdir(parents=True, exist_ok=True)
    records = [study(seed, nodes, ("irregular", "branch", "ring")[index % 3],
                     steps, dt, folder / f"seed-{seed}-wave.npz")
               for index, seed in enumerate(seeds)]
    result = {
        "experiment": "geometry_harmonic_cooperation_v1",
        "seeds": list(map(int, seeds)), "nodes": nodes,
        "steps": steps, "dt": dt,
        "results": records,
        "limits": "Graph scalar oscillators, not elastic plate Chladni modes. "
                  "Drive profile and mode basis are evaluator-only. "
                  "Finite broadband time series yield uncertain spectral peaks. "
                  "This experiment does not train a model or establish qualia, "
                  "consciousness, active sensing, or learned modal cooperation. "
                  "Train the shared local JAX model separately using "
                  "geometry_automata --medium harmonic."
    }
    (folder / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="runs/geometry-harmonics")
    p.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 41])
    p.add_argument("--nodes", type=int, default=16)
    p.add_argument("--steps", type=int, default=720)
    p.add_argument("--dt", type=float, default=.055)
    a = p.parse_args()
    result = run(a.out, tuple(a.seeds), a.nodes, a.steps, a.dt)
    print(json.dumps({"seeds": result["seeds"],
                      "out": a.out}, indent=2))


if __name__ == "__main__":
    main()
