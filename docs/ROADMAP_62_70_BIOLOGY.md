# Parallel biological branch — roadmap 62–70

Goal: determine whether *measured* connectivity and dynamics provide reproducible computational constraints. Preserve this branch independently of the RGB demonstration.

| Roadmap | Question | Evidence boundary |
|---|---|---|
| 62 | Can we ingest genuine directed connectomes? | Provenance contract, SHA, licensing and synthetic source declaration |
| 63 | How does a graph register to retinotopic space? | Requires *supplied* anatomical coordinates; synthetic geometry != anatomy |
| 64 | Do graph constraints affect learned prediction? | Three-coefficient frozen graph/rewire/no-message controls under a synthetic teacher |
| 65 | Are prediction pathways causally necessary? | Frozen model lesions vs matched node controls, not biological injury experiments |
| 66 | Do visual stimuli predict neural responses? | Optional external session-held-out neural recordings, otherwise synthetic only |
| 67 | Which graph harmonic modes matter causally? | Explicitly symmetrized Laplacian, energy-matched mode interventions; not brain waves |
| 68 | How do grid/graph/particle message routes compare? | Identical synthetic node signals, per-edge cost proxy, no measured device energy |
| 69 | Do local rules transfer between graphs? | Toy-graph zero-shot coefficient transfer vs target-supervised reference |
| 70 | Are claims and artifacts independently verifiable? | Evidence ledger refuses to promote synthetic results |

Run `python -m experiments.roadmap_70_biology` after merging the PR stack. Expect `synthetic_controls_only` and no supported biological claims. This is **correct behavior**, not failure: neither a connectome measurement nor neural recordings have been provided in the current repository. A future real-data campaign must: identify compatible neural and connectomic datasets (species, circuit, preparation and stimulation), validate timestamps and anatomical node mapping, freeze controls and noise-ceiling protocol before test, document license and hashes, and separate measured data from any simulated graph/particle visual substrate comparisons.

Scientific reporting must not confuse graph Laplacian modes with physical oscillatory waves, or model lesions with interventions on actual neural tissue. Reuse conclusions for other world models only when hypotheses survive the appropriate held-out domain and causal-controls.
