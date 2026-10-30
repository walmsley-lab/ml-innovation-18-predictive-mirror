# PR 19 — Frozen campaign, no automatic GPU spending

`experiments/campaign.json` declares a pilot comparing three models, two curriculum orders and five independent seeds (30 conditions). It fixes the sample budget, loss, replay and **primary** evaluation outcome: genuine barrier reappearance location MAE. All other metrics remain exploratory; report event counts and undefined results.

```bash
python -m campaign                       # print plan ONLY
python -m campaign --execute --max-runs 30 --require-gpu  # operator-approved GPU session ONLY
```

Execution refuses plans exceeding the explicit ceiling and stores a full config hash and source revision with the output; the existing sweep resumes completed conditions. The campaign script does **not** deploy to Akash or cap billing. For a cheaper first run, make a **separate versioned JSON protocol** with fewer seeds; don't rewrite a frozen campaign in place. A 240-update pilot is not evidence of convergence. Verify one-GPU Docker/Akash training and off-provider restore before spending on 30 runs. Compare losses with a separate frozen protocol; don't mix objectives in one contrast.
