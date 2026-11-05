# Calibration and memory: roadmap 31–35

**Purpose:** establish small causal reference experiments for uncertainty,
continual learning, rehearsal, multimodal input and validation-gated curricula.
These are *additive*, independent experiments. None replaces or disqualifies the
CNN, transport, RGB NCA, hierarchical, Gaussian, graph or particle families.

Start with the test suite:

```bash
python -m unittest tests.test_roadmap_31_calibration \
    tests.test_roadmap_32_adaptation tests.test_roadmap_33_replay \
    tests.test_roadmap_34_crossmodal tests.test_roadmap_35_gates \
    tests.test_roadmap_31_rgb_calibration tests.test_roadmap_32_rgb_adaptation \
    tests.test_roadmap_33_rgb_replay tests.test_roadmap_35_rgb_gates -v
```

| Roadmap | Runner | Question and relevant controls | Current evidence level |
|---|---|---|---|
| [31](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/133) | `python -m experiments.roadmap_31_calibration` and `python -m experiments.roadmap_31_rgb_calibration` | Do observed volatility bins improve interval sharpness without breaking coverage? Global conformal/fixed width | Scalar calibration reference plus actual frozen RGB forecast *pixel-bound* wrapper; neither learns uncertainty |
| [32](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/134) | `python -m experiments.roadmap_32_adaptation` and `python -m experiments.roadmap_32_rgb_adaptation` | Can post-observation updates recover from A→B→A shifts? Frozen and stale-feedback controls | Feature-regression control plus causal post-score SGD on the real RGB NCA; independent held-out video remains untested |
| [33](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/135) | `python -m experiments.roadmap_33_replay` and `python -m experiments.roadmap_33_rgb_replay` | Does bounded earlier-example rehearsal change forgetting and new-task fit at equal updates? Recent-only control | Scalar control plus full recurrent RGB NCA rehearsal under matched updates; no demonstrated device-cost advantage |
| [34](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/136) | `python -m experiments.roadmap_34_crossmodal` | Does RGB plus timestamp-aligned inertial signal improve motion prediction and survive missing channels? Vision, inertial, shuffle, dropout, video coasting | Synthetic single-axis rendered RGB + noisy motion-sensor proxy, not actual IMU |
| [35](https://github.com/walmsley-lab/ml-innovation-18-predictive-mirror/pull/137) | `python -m experiments.roadmap_35_gates` and `python -m experiments.roadmap_35_rgb_gates` | Can validation-only capability gates schedule training without test leakage? Fixed and mixed exposure at equal updates | Toy reference plus RGB NCA held-out domain gates; **not** full charter capability certification |

## What these implementations do *not* establish

A passing unit test says that scheduling, provenance, arithmetic and causality
constraints function. It does **not** establish trained network superiority,
real-camera transfer, biological relevance, calibrated 2D video distributions or
on-device efficiency. Report those separately after substantive training and
untouched evaluation. In particular, stage labels in Roadmap 35 are toy task
complexity levels, *not* completion of the
[developmental capability charter](DEVELOPMENTAL_CAPABILITIES.md).

## Next integration gates

1. **Freeze the real task and camera domain.** Use the existing RGB motion,
   layered-world and independently recorded camera protocols. Preserve source
   seeds, image resolution and timestamp semantics; predeclare uncertainty
   coverage/score and domain-shift definitions.
2. **Reuse identical inputs and trained checkpoints.** Wrap frozen CNN,
   transport, recurrent RGB, hierarchical and Gaussian forecasts with the same
   no-future calibration protocol. Keep independent *whole episode/recording*
   splits for training, calibration, validation, and held-out test. Calibrate
   categorical multi-future probabilities with proper scores, not optimistic
   best-of-many pixel error.
3. **For online adaptation, commit before learning.** A frame may update a
   model *only after* its original forecast has been scored. Track optimizer
   steps, stale/missing labels, delayed feedback, CPU/GPU time and parameter
   changes. Reset experiments only on allowed observables.
4. **Carry replay into the actual recurrent models.** Match gradient count,
   examples, parameter capacity and inference budget. Record replay memory
   bytes and reads; compare forgetting on old domains *and* acquisition on new
   domains. Memory is a cost, not a free resource.
5. **Use real aligned modalities.** Replace the inertial proxy only when
   sensor/video synchronization, latency and missing-signal semantics are
   measured. Test sensor shuffles and held-out camera actions, and record
   per-modality failures rather than treating more channels as inherently
   better.
6. **Gate with actual visual competencies.** Replace Roadmap 35's toy probes
   with the frozen motion/occlusion/viewpoint/uncertainty/attention measures in
   `experiments/developmental_capabilities.json`. Thresholds and comparator
   choices must be fixed before evaluating untouched scenes, with separate
   validation and confirmatory test records. Report nonpromotion, late
   promotion and schedule-specific compute alongside any accuracy gain.

**Policy:** a poor early result is a measured datapoint at a particular
resolution, dataset, optimizer, budget and substrate. Keep architecture
variants, strong classical baselines and seed-level curves for future
reevaluation; do not turn one small CPU pilot into an elimination decision.
