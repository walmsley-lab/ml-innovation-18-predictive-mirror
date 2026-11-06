# Picbreeder-inspired open-ended embodied curriculum

**Idea:** Reuse Picbreeder's branching, interactive evolution of compact
generative programs to supply *varied physical worlds* for our sensorimotor
learner. Preserve a clean distinction between the **environment generator**
and the **agent trained to perceive it**.

## Why this belongs here

Picbreeder (Secretan et al.; Stanley and colleagues) uses compositional
pattern-producing networks (CPPNs), evolved with NEAT-like mutations and
human selection, to generate images. The key insight is not a better
single-object objective; it is branching exploration of a vast design space,
where a new line of investigation can begin from any previous ancestor.

Our embodied curriculum already simulates springs, contact and a simple
fivefold sand-dollar-like outline. Its current procedural distributions are
too narrow to support strong generalization claims. Evolving or branching
generative shape programs would let us explore irregular boundaries, cavities,
texture, stiffness and resonance while preserving an exact causal simulator.

## Research progression (keep independent from the reference trainer)

1. **Compact CPPN shape generator.** Map coordinates `(x, y, radius, bias)`
   to occupancy, height or compliance, with seeded reproducibility. Save the
   genotype, generation and ancestor IDs, not just a rendered image.
2. **Branching selection.** Allow a human to select interesting valid shapes
   and spawn small mutations. Preserve an unselected random lineage as a
   novelty/coverage reference. Include mutation magnitude and viable-geometry
   constraints; don't optimize for recognizable images alone.
3. **Causal sensing.** Sample fingertip traces, indentation/contact,
   proprioception and vibrations from *the same generated physical object*.
   A 2D texture by itself is not a tactile material or a mechanical model.
4. **Matched transfer.** Train identical models on fixed handcrafted shapes,
   equally sized random CPPNs, and selected evolving lineages. Evaluate on
   lineage-disjoint shapes and untouched contact locations, including
   eyes-closed trials.
5. **Optional body evolution.** Only after shape transfer is demonstrated,
   explore actuator placement and tactile sensor geometry under explicit
   budgets. Keep body design separate from neural model learning.

**Primary question:** Does a branching diversity curriculum improve
object understanding and generalization compared with a straightforward
randomized curriculum at matched training examples and compute?

## Guardrails

- Avoid target-shape leakage: split by **ancestor/lineage**, not only image.
- Require valid topology and stable contact mechanics, not just visual appeal.
- For any sensory task, evaluate against geometric and Kalman/filtering
  baselines as appropriate, not only another NCA.
- Store genotype, seed, mutations, selected parents, sensing actions and
  rendering parameters in an immutable manifest.
- Keep human preferences as selection metadata; they are not ground truth
  that a candidate has learned perception or developed sentience.
- Start with a minimal CPU implementation; do not add paid evolutionary
  search, a UI framework or NEAT dependency before controls are ready.

## References

- Original Picbreeder: https://picbreeder.net/
- Secretan et al., *Picbreeder: A Case Study in Collaborative Evolutionary
  Exploration of Design Space* (2011).
- Sakana AI's 2026 reproduction using VLM-driven selection:
  https://pub.sakana.ai/picbreeder-vlm/

**Status:** Research note and proposed next step only. The current
`embodied_causal_curriculum.py` still trains against the original synthetic
worlds. No CPPN evolution, interactive selector or improved transfer
performance is claimed in this PR.
