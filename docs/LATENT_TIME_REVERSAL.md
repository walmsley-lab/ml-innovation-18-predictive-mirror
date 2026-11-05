# Forward/backward inference and sufficient predictive memory

**Question.** Does maintaining a causal belief over hidden states improve
future prediction over simply using the newest image/observation?

```bash
python -m experiments.latent_forward_backward --episodes 200 --steps 20 --seed 0
python -m unittest tests.test_latent_forward_backward -v
```

## Mechanism

An explicitly known three-state cyclic hidden Markov process emits noisy binary
observations. **Filtering** computes P(z_t | y_0:t), updating the internal belief
using the transition kernel and evidence seen by t. **Smoothing** computes
P(z_t | y_0:T) with a backward pass and uses future observations; it is included
for retrospective latent-state diagnosis ONLY. The stationary time-reversed
kernel R(current,previous) follows Bayes, not T inverse, and is a separate
mathematical object from the HMM backward message.

## Controls and reported evidence

- Evaluate next-observation log loss on fixed seeded independent episodes:
  full history, newest observation only, temporally shuffled prior observations
  (latest retained), and unconditional stationary forecast.
- No predictor sees y[t+1] before issuing its forecast. Synthetic hidden states,
  emission probabilities, and transition probabilities are **oracle-known**,
  rather than inferred from pixels. Smoothing is deliberately excluded from
  the forecast scores because it would leak future observations.
- Report mean per-forecast negative log likelihood, number of forecasts,
  seed, and absolute values. Do not label an NLL improvement as recognition,
  object permanence, a learned world model, or thermodynamic dissipation.
- This HMM is a unit test of the *inference question*, not a trained
  Predictive Mirror result. Because the transition model is known, the
  filtered belief is sufficient for all future predictions under this model.

## Promotion to RGB NCA (separate work)

1. Freeze a trained RGB checkpoint; evaluate on independent scenes and sessions.
2. Compare a causal latent-state predictor against equal-capacity truncated
   history, input-only, shuffled hidden-state, persistence, and transport
   controls. Measure +1/+3/+5/+10 horizons, occlusion recovery and calibration.
3. Precommit forecasts and audit no future RGB/visibility/flow truth enters
   predictors. Restrict retrospective smoothing to analysis after outcomes.
4. Quantify whether the hidden state actually carries predictive information
   beyond recent observations, with paired uncertainty across held-out episodes
   and independently trained seeds. Failure is a legitimate result.
5. Do not estimate a reverse-time kernel from high-dimensional continuous
   hidden states without explicit density assumptions, stationarity checks,
   and a validity study of the coarse-graining.

## References

- Rabiner (1989), HMM forward–backward inference,
  https://www.cs.cmu.edu/~durand/03-711/Readings/Rabiner89.pdf
- Littman, Sutton & Singh (2001), *Predictive Representations of State*,
  https://proceedings.neurips.cc/paper_files/paper/2001/hash/1e4d36177d71bbb3558e43af9577d70e-Abstract.html
- Seifert (2012), stochastic time reversal,
  https://arxiv.org/abs/1205.4176

**Status:** exact toy model and tests; not proof of useful learned memory.
