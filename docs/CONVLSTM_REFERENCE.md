# Small conventional recurrent video predictor

Reference: Shi et al. (2015), [ConvLSTM](https://arxiv.org/abs/1506.04214). This PR implements a **minimal one-layer JAX ConvLSTM**, not the paper's full radar-nowcasting experiment. It is a performance baseline for the *same* RGB paired-history task, not a claim of novelty.

```bash
python -m unittest tests.test_convlstm_reference -v
python -m experiments.convlstm_paired_baseline --steps 100 --seeds 11
```

The training loop sees only past RGB context; supervised future frames enter the loss. Model outputs have shape `[B,H,Hpx,Wpx,3]`. Independent validation scene seeds select what training learned. A frozen test comparison and capacity/training-budget parity against the cellular NCA are subsequent requirements.

For reference-*faithful* SimVP, use the existing framework-neutral paired-video export (research PR #193) with a pinned upstream OpenSTL/SimVP checkout. Do not substitute this small ConvLSTM as a SimVP reproduction or compare published Moving-MNIST MSE with our normalized RGB task. Keep modern external reference environments separate from the JAX core.
