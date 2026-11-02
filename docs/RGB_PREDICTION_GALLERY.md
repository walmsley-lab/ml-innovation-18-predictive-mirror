# PR 46 — Frozen RGB one-second prediction gallery

This is a viewer and measurement tool, **not** another trained model. It
loads an existing RGB NCA checkpoint and evaluates the identical synthetic
RGB episode against persistence and observed-image constant velocity.

Per horizon (+1/+3/+5/+10 frames), labeled PNG columns show the last
observed frame, NCA forecast, persistence, constant velocity, actual
future frame, absolute NCA error, and diagnostic NCA cell activity.

    python -m experiments.rgb_prediction_gallery \
      --parameters runs/rgb-nca-1s/parameters.npz \
      --seed 71000 --episodes 3 --context 8 --horizon 10 \
      --out runs/rgb-prediction-gallery

Each PNG and metrics.json are generated under the output folder. A
checkpoint SHA256 is recorded. Pillow is optional for the rest of the repo
but necessary to generate gallery PNGs. No camera required.

The model sees ONLY the observed context; future frames are accessed
after predictions for scoring and labeled visual comparison. Scene seeds
are *not* automatically proven held out; exclude them from training and
checkpoint selection before interpreting generalization scores.

The internal activity colors are a visualization of a scalar hidden-state
summary, not an object representation or biological brain measurement.
