# PR 10 — Hidden-Rules World calibration fixture

Minimal controllable world: constant/drag/acceleration rules, an occluder
and known camera actions, returning separate model observations and privileged
positions/rule. Rule variants share the same **pre-switch** visible history;
the hidden rule becomes distinguishable only after its effect is observed.

Follow-on: add strong Bayesian/oracle/reactive calibration, held-out physics
ranges, collisions, object combinations and world-difficulty gates. Current
code is a simulator fixture, not a trained model or a claim of physical insight.
