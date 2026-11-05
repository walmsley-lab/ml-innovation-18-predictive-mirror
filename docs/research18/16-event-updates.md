# PR 16 — Actual selective-cell computation

Cell updates are *not executed* for inactive sites, and cost accounting
records reads, multiplies and updated fraction. Dense/all-active/none
controls have known answers. No claim that sparse Python execution is faster
than vectorized JAX or that energy was measured.

Next: combine with trained recurrent rules, compute matched random masks
and prediction-error masks using **past evidence only**, and evaluate
iso-accuracy communication and wall-clock cost. Include mask generation,
indexing, data movement and delay overhead when reporting efficiency.
