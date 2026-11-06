/* Pure, deterministic geometry for voluntary audiovisual scene sketches.
 * No camera, microphone, network, or model inference is involved. */
"use strict";

const PerceptualAtlas = (() => {
  const FORMAT = "perceptual-atlas-v1";
  const MODES = ["spiral", "fold", "lattice", "chamber"];
  const DEFAULT = Object.freeze({
    seed: 7, mode: "spiral", light: 0.15, detail: 0.55,
    flow: 0.42, pitch: 0.42, coupling: 0.65
  });
  const clamp = (value, lower, upper, fallback) => {
    if (value === null || value === undefined || value === "") return fallback;
    const number = Number(value);
    return Number.isFinite(number)
      ? Math.max(lower, Math.min(upper, number)) : fallback;
  };
  const frac = (x) => x - Math.floor(x);
  const hash = (n) => frac(Math.sin(n * 127.1 + 39.7) * 43758.5453);

  function normalizeScene(input = {}) {
    const value = input && typeof input === "object" ? input : {};
    return {
      seed: Math.round(clamp(value.seed, 0, 999999, DEFAULT.seed)),
      mode: MODES.includes(value.mode) ? value.mode : DEFAULT.mode,
      light: clamp(value.light, 0, 1, DEFAULT.light),
      detail: clamp(value.detail, 0, 1, DEFAULT.detail),
      flow: clamp(value.flow, 0, 1, DEFAULT.flow),
      pitch: clamp(value.pitch, 0, 1, DEFAULT.pitch),
      coupling: clamp(value.coupling, 0, 1, DEFAULT.coupling),
    };
  }

  function normalizeSketch(input = {}) {
    if (input.format && input.format !== FORMAT) throw new Error("Unsupported sketch format");
    const initial = normalizeScene(input.initial || DEFAULT);
    const rawEvents = Array.isArray(input.events) ? input.events.slice(0, 512) : [];
    const events = rawEvents.map((event) => ({
      t: clamp(event && event.t, 0, 900, 0),
      scene: normalizeScene(event && event.scene),
    })).sort((a, b) => a.t - b.t);
    const duration = clamp(input.duration, 0, 900,
      events.length ? events[events.length - 1].t : 0);
    return {format: FORMAT, initial, events, duration: Math.max(duration,
      events.length ? events[events.length - 1].t : 0)};
  }

  function sceneAt(sketch, seconds) {
    const time = Math.max(0, Number(seconds) || 0);
    let scene = sketch.initial;
    for (const event of sketch.events) {
      if (event.t > time) break;
      scene = event.scene;
    }
    return scene;
  }

  // A compact generative toy world. Lighting changes presentation but not
  // the underlying spatial field; frequency couples synthetic tone to shape.
  function field(x, y, seconds, input) {
    const s = normalizeScene(input);
    const radius = Math.hypot(x, y);
    const angle = Math.atan2(y, x);
    const t = Math.max(0, seconds) * (0.14 + 0.64 * s.flow);
    const seedPhase = hash(s.seed) * Math.PI * 2;
    const density = 5 + Math.round(s.detail * 11);
    const pitchPhase = (1.2 + 7 * s.pitch) * s.coupling;
    const theta = angle + 0.23 * Math.sin(radius * 3 - t + seedPhase);
    let a, b;
    switch (s.mode) {
      case "fold":
        a = Math.cos(density * theta + 2.6 * Math.sin(4 * radius - t));
        b = Math.cos(16 * radius - 1.5 * t + pitchPhase * theta);
        break;
      case "lattice":
        a = Math.cos(density * (x + 0.25 * Math.sin(y * 5 + t)));
        b = Math.cos(density * (y - 0.25 * Math.sin(x * 5 - t)) + pitchPhase);
        break;
      case "chamber":
        a = Math.cos(density * Math.log(radius + 0.12) - t + seedPhase);
        b = Math.cos((density / 2) * theta + 3 * Math.sin(3 * radius - t));
        break;
      default:
        a = Math.sin(density * theta / 2 + 7 * Math.log(radius + 0.10) - t);
        b = Math.cos(19 * radius - 1.3 * t + pitchPhase * Math.sin(theta * 3));
    }
    const interference = Math.sin(4 * radius + 2 * theta - 0.7 * t + seedPhase);
    return {
      intensity: Math.max(0, Math.min(1, 0.5 + 0.26 * a + 0.19 * b + 0.05 * interference)),
      hue: 0.5 + 0.5 * Math.cos(theta * 2 + radius * 4 - 0.3 * t + seedPhase),
    };
  }

  function rgb(x, y, seconds, scene) {
    const s = normalizeScene(scene);
    const f = field(x, y, seconds, s);
    const wave = f.intensity;
    const accents = [
      0.50 + 0.50 * Math.cos(6.283 * f.hue),
      0.50 + 0.50 * Math.cos(6.283 * f.hue + 2.09),
      0.50 + 0.50 * Math.cos(6.283 * f.hue + 4.18),
    ];
    const base = 10 + s.light * 234;
    const strength = (0.13 + 0.87 * wave) * (0.35 + 0.65 * Math.abs(2 * wave - 1));
    return accents.map((component) => Math.round(Math.max(0, Math.min(255,
      s.light < 0.5
        ? base + (1 - s.light) * 195 * strength * (0.32 + 0.68 * component)
        : base - s.light * 172 * strength * (0.32 + 0.68 * component)
    ))));
  }

  function toneHz(scene) {
    const s = normalizeScene(scene);
    return 110 * Math.pow(2, s.pitch * 1.5);
  }

  return {FORMAT, MODES, DEFAULT, normalizeScene, normalizeSketch,
    sceneAt, field, rgb, toneHz};
})();

if (typeof module !== "undefined" && module.exports) module.exports = PerceptualAtlas;
