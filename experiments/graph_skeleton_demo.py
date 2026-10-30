"""Optional animated skeleton evaluation; future truth is display-only."""

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from experiments.graph_skeleton_nca import episode
from src import graph_skeleton_nca as nca


def render(checkpoint, output, seed=50_041_000, occlusion_frames=3):
    from PIL import Image, ImageDraw

    with np.load(checkpoint, allow_pickle=False) as f:
        params = {name: jnp.asarray(f[name]) for name in f.files}
    observed, visible, truth = episode(seed, occlusion_frames=occlusion_frames)
    forecast = np.asarray(nca.predict(params, jnp.asarray(observed[None]),
                                      jnp.asarray(visible[None]), nca.adjacency()))[0]
    labels = ("Observed at t", "Predicted at t+1", "Truth at t+1 (evaluation)")
    colors = ((65, 215, 212), (251, 171, 70), (190, 205, 216))
    images = []
    for t, predicted in enumerate(forecast):
        image = Image.new("RGB", (690, 245), (16, 27, 39))
        draw = ImageDraw.Draw(image)
        for i, (label, pose, color) in enumerate(zip(
                labels, (observed[t], predicted, truth[t+1]), colors)):
            offset = i * 230
            draw.text((offset+10, 10), label, fill=(235, 242, 247))
            def xy(node):
                return (offset + 12 + int(205 * pose[node, 0]),
                        24 + int(205 * pose[node, 1]))
            for a, b in nca.EDGES:
                if i == 0 and (visible[t, a, 0] == 0 or visible[t, b, 0] == 0):
                    continue
                draw.line((xy(a), xy(b)), fill=color, width=3)
            for node in range(len(nca.JOINTS)):
                if i == 0 and visible[t, node, 0] == 0:
                    continue
                x, y = xy(node)
                draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
        images.append(image)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=150, loop=0)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", default="runs/graph-skeleton-nca/graph.npz")
    parser.add_argument("--out", default="runs/graph-skeleton-nca/demo.gif")
    parser.add_argument("--seed", type=int, default=50_041_000)
    parser.add_argument("--occlusion-frames", type=int, choices=(1, 3, 5), default=3)
    args = parser.parse_args()
    print(render(args.parameters, args.out, args.seed, args.occlusion_frames))


if __name__ == "__main__":
    main()
