# Real singing-bowl resonance: exploratory observation protocol

## Question

Can a water-filled metal singing bowl show a repeatable, abrupt onset of
organized surface waves as vibration amplitude increases? Do observations
support the **edge-induced Faraday-wave** mechanism studied by Terwagne &
Bush (2011), rather than an assumed "coherence" transition?

This is a real-world **measurement protocol**, not a completed experiment.
No frequency or transition threshold is claimed for this particular bowl.
The bowl's engraved motifs must not be conflated with measured elastic
normal modes. The alloy, thickness, and provenance of the instrument are
not independently established.

## Setup (protect the instrument)

- Use room-temperature clean water and the ordinary soft/covered playing
  mallet. Inspect the bowl for cracks, loose finish and an undamaged rim.
  Avoid abrasive mallets, hard striking, electronics in water, and powerful
  external acoustic drive. Stop if anything rattles, cracks, or feels wrong.
- Place the bowl upright on a stable soft pad/ring that leaves the rim
  free. A flat, level surface and repeatable support are essential.
- Record from a fixed phone/tripod angle with bright diffuse or low-angle
  sidelight. Use 120/240-fps video if available, but **do not use frame rate
  alone to infer the true liquid oscillation frequency** (temporal aliasing).
  Record audio alongside video at moderate microphone gain.
- Start empty: capture one gentle tap and one softly sustained rim tone.
  This gives a **dry reference** for comparing frequency shifts after water
  loading. Do not assume the bowl's measured pitch will match literature.
- Add a modest measured amount of water, starting around 5–10 mm above the
  lowest point; use water volumes or an interior depth reference that can
  be repeated. Repeat at one or two greater water depths if the instrument
  permits. Avoid filling close to the rim or risking spills.

## One repeatable trial

1. Begin video and audio before touching the rim. Keep the camera, lighting
   and bowl support fixed throughout each comparison.
2. Rub the rim using the customary smooth technique and fairly steady
   revolution rate. Gradually increase mallet pressure only enough to
   sustain a stable tone. Avoid forceful rubbing or sudden hard impacts.
3. Watch especially near the water-wall contact: initially tiny ripples
   may give way to an organized circumferential standing-wave pattern.
   Record the time when the change first appears, whether the pattern
   persists, any visible spatial nodes, and whether it disappears again
   as the driving weakens.
4. Stop before aggressive splashing or droplet ejection. Let the bowl and
   water settle before the next attempt.
5. Repeat at least three times per water depth and photograph the
   reference fill level. Dry the instrument gently afterward.

## Minimal log (CSV, one row per trial)

- date, trial_id, bowl_id, water_depth_mm (or measured_volume_ml)
- support, mallet_type, lighting, video_path, audio_path
- approximate_rubbing_rate, drive_intensity (low/medium),
  clear_tone (yes/no)
- first_ripples_s, organized_edge_pattern_s, pattern_description
- apparent_azimuthal_nodes, repeatable_transition (yes/no/uncertain)
- relevant recording artifact, interruptions, observations

Use `runs/real-bowl/<date>/` for local captures and `trials.csv`.
Keep raw recordings out of git until privacy, licensing, size and consent
choices have been made. A small **de-identified** summary and protocol
can live in the repository.

## Analysis and controls

- Compare dry and wet sound spectra with a fixed recording placement.
  Audio peaks can approximate vibration **frequencies**, not rim
  displacement or forcing **amplitude** without calibration.
- Mark the water-wave onset on video, aligned with the audio timeline.
  Plot onset versus water level; compare with a same-level low-drive trial
  in which no organized surface pattern appears.
- Repeat after a settling period. Prefer a gradually increasing/decreasing
  drive ramp where practical to look for repeatability and hysteresis.
  Hand-rubbing does **not** keep vibration amplitude constant automatically.
- Differentiate steady capillary ripples, standing edge waves, and
  splashing. Mode selection can change with water depth.
- To test whether the water oscillation is subharmonic (classically near
  half the rim vibration frequency), obtain **independent high-speed
  measurements** or calibrated optical sensing. Ordinary smartphone video
  may alias such frequencies and cannot establish the relation alone.
- Avoid a numerical fit that assigns plate material parameters from one
  sound peak. Begin with clearly labeled phenomenological/graph models
  and only promote a structural model with geometry, calibration and
  held-out measurements.

## Repo boundary

This *protocol and its summarized evidence* live alongside the existing
`src/geometry_harmonics.py` and `experiments/geometry_harmonics.py`
in Predictive Mirror. No modification to the frozen experimental claims
of the separate Developmental Physical Computing nanowire research program
is warranted. A new physical-resonance repository is only justified if the
collection grows into multiple instruments, custom hardware or a distinct
measurement library.

## Source

Terwagne, D. & Bush, J. W. M. (2011), *Tibetan singing bowls*,
Nonlinearity 24 R51–R66; DOI: 10.1088/0951-7715/24/8/R01.
Open MIT research record:
https://dspace.mit.edu/entities/publication/9539bdc0-5328-40b8-88e6-02518c796e49

The paper documents rim vibration, associated surface waves, edge-induced
Faraday-wave onset and droplet generation as driving intensity grows.
Any claimed observation for this bowl remains to be measured.
