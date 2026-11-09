"""Evaluate whole-state swaps on a frozen RGB NCA checkpoint."""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from experiments.object_permanence_campaign import dataset, episode_success
from src import predictive_rgb_nca as nca


def evaluate(params, records, observations, horizon, ticks=2):
    """Swap state after the last observed frame, then forecast open loop."""
    if len(records) != len(observations) or not len(records) or len(records) % 2:
        raise ValueError("Need complete ordered episode pairs")
    if not np.array_equal(observations[::2, -1], observations[1::2, -1]):
        raise ValueError("Paired final observations must be identical")
    if any(record["visibility_reason"][observations.shape[1] - 1] != "barrier"
           for record in records):
        raise ValueError("Swap requires complete occlusion")
    if ticks < 1 or horizon < 1:
        raise ValueError("Need positive horizon and ticks")

    context = observations.shape[1]
    observed = jnp.asarray(observations)
    state = nca.encode_context(params, observed, ticks)
    order = np.arange(len(records))
    donor = order ^ 1

    def forecast(indices):
        patched = jax.tree.map(lambda leaf: leaf[indices], state)
        _, frames = jax.lax.scan(
            lambda carry, _: nca.dream(params, carry, ticks),
            patched, None, length=horizon)
        return np.asarray(jnp.swapaxes(frames, 0, 1))

    intact = np.asarray(nca.rollout(params, observed, horizon, ticks))
    copied = forecast(order)
    swapped = forecast(donor)
    np.testing.assert_allclose(copied, intact, rtol=1e-6, atol=1e-7)

    comparisons = {
        "intact": episode_success(intact, records, context),
        "same_history_copy": episode_success(copied, records, context),
        "swapped_vs_recipient": episode_success(swapped, records, context),
        "swapped_vs_donor": episode_success(
            swapped, [records[i] for i in donor], context),
    }
    pairs = []
    for i in range(0, len(records), 2):
        directions = slice(i, i + 2)
        eligible = all(comparisons["intact"]["success_per_event"][directions])
        follows_donor = (
            all(comparisons["swapped_vs_donor"]["success_per_event"][directions])
            and not any(comparisons["swapped_vs_recipient"]["success_per_event"][directions])
        )
        pairs.append({"seed": records[i]["seed"], "intact_pair_passed": eligible,
                      "bidirectional_switch": follows_donor if eligible else None})
    eligible_count = sum(pair["intact_pair_passed"] for pair in pairs)
    switched = sum(pair["bidirectional_switch"] is True for pair in pairs)
    return {
        "protocol": "paired-rgb-nca-state-swap-v1",
        "context": context, "horizon": horizon, "ticks": ticks,
        "comparisons": comparisons, "pairs": pairs,
        "eligible_pairs": eligible_count, "switched_pairs": switched,
        "switch_rate": switched / eligible_count if eligible_count else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/state-swap/metrics.json"))
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--ticks", type=int, default=2)
    args = parser.parse_args()

    records, frames = dataset("test", args.pairs, args.context, args.horizon, args.size)
    with np.load(args.checkpoint, allow_pickle=False) as archive:
        params = {name: jnp.asarray(archive[name]) for name in archive.files}
    report = evaluate(params, records, frames[:, :args.context], args.horizon, args.ticks)
    report["checkpoint"] = str(args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f'{report["switched_pairs"]}/{report["eligible_pairs"]} eligible pairs switched '
          f'({len(report["pairs"])} pairs evaluated)')


if __name__ == "__main__":
    main()
