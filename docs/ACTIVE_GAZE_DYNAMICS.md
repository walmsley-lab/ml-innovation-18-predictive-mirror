# PR 53 — Active gaze in a small moving-object world

A 1D target moves and reflects off world boundaries. A camera only sees an
8-pixel-wide local window, so its actions control which evidence arrives.
We compare **fixed**, **random**, and information-seeking **active** gaze
policies. The active policy uses only its current belief over position and
velocity, and chooses a legal camera step that minimizes expected posterior
future-position variance. It never reads the simulator's true target state.

Every policy also has two matched estimators: action-aware (knows the
camera's actual viewing window) and action-blind (incorrectly assumes the
camera never moved, resetting when its hypothesis becomes impossible).
Both receive identical actual local measurements from each policy.

    python -m experiments.active_gaze_dynamics \
      --seeds 11 23 37 --episodes 24 --context 8 --horizon 10
    python -m unittest tests.test_active_gaze_dynamics -v

Output includes future position error, forecast entropy and example gaze
traces. There are no learned weights, no NCA coupling, and no real RGB
video in this PR. The dynamics and sensor model are KNOWN to this Bayesian
controller. It is a controlled action-perception baseline, not evidence
of self-organized neural attention. The next experiment can test an
NCA world model under the same action and observation contract.
