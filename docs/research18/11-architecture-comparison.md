# PR 11 — Fair cross-architecture evaluator

Reusable paired evaluation on identical **visible frames**, same contexts,
horizons and per-lead pixel scores. Includes persistence and image-difference
extrapolation as reference controls. No trained NCA/CNN/SSM comparisons are
claimed by this initial evaluator. Such models must be trained under PR 02/04
and scored with unchanged input privileges, explicit parameter/cost axes,
held-out world configurations and paired independent episodes.

Keep oracle skeleton/particle state comparisons in separately labeled groups.
