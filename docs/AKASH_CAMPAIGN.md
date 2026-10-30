# PR 25 — Last Akash campaign packaging gate

The Docker image now includes the preregistered campaign runner and has three explicit entry modes: RUN_MODE=train (default), RUN_MODE=sweep, RUN_MODE=campaign. The campaign mode fails *before the public artifact server starts* unless CAMPAIGN_APPROVED=1. It also refuses to start more than CAMPAIGN_MAX_RUNS (defaults to 1), while the frozen pilot manifest plans 30 separate training runs. STEPS/BATCH_SIZE in the ordinary Akash SDL apply to train/sweep; campaign mode reads experiments/campaign.json instead.

The first GPU smoke must use RUN_MODE=train, STEPS=3, BATCH_SIZE=2, and a verified immutable GHCR digest. Verify JAX GPU device, restore a checkpoint off-provider, close the lease, and **read back closed status** before attempting a campaign.

Only after that, if approved, set RUN_MODE=campaign, CAMPAIGN_APPROVED=1 and CAMPAIGN_MAX_RUNS=30 and check the frozen JSON and price/bid. This is not a spending cap; a lease can continue billing even if the script stops. Do not treat the SDL price bid or escrow as a hard time-based budget.

CI checks shell syntax and read-only campaign planning; it does not start paid compute.
