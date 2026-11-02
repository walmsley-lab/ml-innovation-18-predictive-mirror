# PR 50 — Content-conditioned soft reference selection

A new, separately trained RGB NCA whose cells select information through a
weighted mixture of **nine fixed relative offset candidates**, up to six
pixels away. The weights are learned from each cell's hidden state and
current image; this is **not** hard pointers to recognized objects or
unbounded global connectivity.

Paired modes use the same initial weights, scenes and training updates:
uniform selection, learned soft selection and shuffled correspondence
(identical learned weight values reassigned to other offsets).
All save separate checkpoints and compare with image-only persistence and
constant velocity at +1/+3/+5/+10 frames. Step-zero initialization
remains eligible; softmax routing entropy is logged.

    python -m experiments.rgb_soft_references --steps 40 --horizon 10
    python -m unittest tests.test_rgb_soft_references -v

Different routing mechanisms have different active gradient paths and
effective receptive fields. This ablation cannot establish a general
performance winner without replication and measured compute. Original
fixed local/relative/diagonal-mirror selector models are unchanged.
