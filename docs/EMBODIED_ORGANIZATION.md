# Embodied Organization: learning about what can fit

**Question:** Does a small budget of physical interaction help an agent
organize unfamiliar objects more effectively than visual estimates alone?
Do targeted probes outperform randomly chosen probes under the same budget?

This experiment is a deliberately modest **partially observed constraint
world**. It builds a classical baseline before attempting learned contact
dynamics, neural cellular representations, category-theoretic compositional
planning, or physical robot manipulation.

## World

Each episode has 2–7 containers, each with two hidden dimensions (width and
depth) and a visually supplied profile (round or square). The observer gets
noisy measurements of dimensions, not the true dimensions. A container can
nest inside another exactly when its profile matches and both dimensions
leave a fixed 0.055 clearance. This relation is directed.

An agent can:
- **Observe** the noisy dimensions and supplied profile.
- **Probe** a proposed outer/inner pair to obtain a true yes/no compatibility
  result, representing a bounded tactile test (not a force/torque sensor).
- **Plan** an arrangement as ordered nesting chains (outer to inner).
- **Execute** proposed joins; a failed join yields feedback and the affected
  object starts a new stack. This is a simple local recovery, not global
  replanning.

The objective is the *fewest valid stacks*, with every object used exactly
once. Since the model counts are small, the exact minimum chain cover can
be computed by enumerating simple directed paths and dynamic programming
over object subsets. This is a classical combinatorial planner, not a
trained neural system.

## Four matched comparison arms

| Arm | Source of relation information | Probes |
|---|---|---|
| Vision | Noisy observed dimensions | 0 |
| Random | Vision plus uniformly sampled directed pair checks | At most K |
| Active | Vision plus visually uncertain joins in the current plan | At most K |
| Oracle | Hidden true dimensions | 0; evaluator-only upper bound |

The random and active arms use the same initial world and the same query
budget; each query returns the same truth function. The oracle is not a
fair input-matched learner, only the optimal reference. A probe is extra
information; comparisons against vision-only therefore measure the value
of interaction, not equal-observation learning performance.

The active policy is simple and hand-authored: it recomputes a proposed
plan and tests an unprobed predicted join closest to a visual dimension
threshold. If no such join exists, it selects another uncertain compatible-
profile pair. It does not fit a new rule or transfer learning between episodes.

## Run

Requires only Python standard library; a CPU is sufficient.

\`\`\`bash
python -m unittest tests.test_embodied_organization -v

python -m experiments.embodied_organization \
  --seed 11 --episodes 40 --count 6 --noise 0.11 --budget 3 \
  --out runs/embodied-organization/report.json
\`\`\`

The JSON records episode IDs, visible observations, probe results,
proposed and executed stacks, failed join counts, optimal stack counts and
aggregate paired differences. The output file is not overwritten.

Useful comparisons: mean **excess stacks** over oracle; mean **failed
joins**; **optimal and valid completion** rate; probe counts; active-minus-
random paired excess stack count. Negative differences favor active
probes. Don't assign statistical significance to tiny seed sweeps.

## What the experiment can and cannot establish

It can show whether interaction feedback changes feasible arrangements
in a known, controlled family of constraint problems. It cannot establish
a learned affordance representation, motion prediction, general-purpose
manipulation, human perception, or bodily understanding. The vision input
already contains supplied shape names and numeric features. No images,
pixels, contact forces, continuous rigid-body dynamics, or neural
training are involved. Observations are generated, not collected.

The current nesting rules form a partial order within each object profile.
Treating legal actions as composable transformations is a useful
mathematical viewpoint, but this implementation is not a general category
theory library or proof about human conceptual categories.

## Next gates if the baseline is informative

1. Establish empirical differences over several fresh paired episode
   seeds, including a zero-noise sanity check and probe-cost tradeoff.
2. Learn pairwise compatibility from interactions on **training objects**,
   then freeze and evaluate on disjoint objects, profiles and dimension
   ranges without evaluator-only truth leaking into the representation.
3. Replace given numeric measurements with rendered visual input and
   contact events; compare against the same symbolic baselines.
4. Add nonlocal support/load constraints where valid local fits alone
   do not guarantee a valid complete arrangement. Test repair and
   action composition, rather than just static pairwise fit.
5. Only then compare a small recurrent or cellular rule to the classical
   relation planner on predictive accuracy, generalization and cost.

The lab goal is to understand **which extra structure is earned by
interaction**, not to equate elaborate simulations with intelligence.
