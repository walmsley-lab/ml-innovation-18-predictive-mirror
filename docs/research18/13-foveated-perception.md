# PR 13 — Predictive foveation fixture

Full-resolution source, truly downsampled peripheral sensor, coarse-only gaze
selection and a high-resolution foveal crop. Disentangle **sensing** from
**processing**: the source camera's full-frame capture is accounted separately,
and in-sensor selective readout is NOT assumed. Selector cost must be charged.

Follow-up: frozen, learned predictors and matched budgets; best-fixed gaze
chosen on TRAIN, random, surprise, salience, information-seeking and oracle
reference. Headline criteria include peripheral-event miss rate, long-horizon
prediction accuracy, acquisition cost and total computation. No accuracy
or hardware efficiency win is claimed by these sensor unit tests.
