# Roadmap 40 — irregular frame intervals

Fit the *same-size* linear motion-head reference using RGB-derived subpixel centroids with either an explicitly planned t→t+1 interval or a timing-blind feature. Compare against interval-normalized velocity, per-frame velocity and persistence. Full independent episodes separate train/validation/test. Test regular cadence, variable cadence, dropped scheduled frames and acceleration.

**Timing contract:** planned next observation interval is known at forecast issuance; actual subsequent capture timestamp is not. Do not pretend this matches an uncontrolled webcam without capture planning or latency instrumentation. This is a one-step classical RGB feature study, not yet a continuous-time NCA.
