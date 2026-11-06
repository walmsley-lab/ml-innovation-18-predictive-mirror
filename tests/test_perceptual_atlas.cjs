"use strict";
const assert = require("node:assert/strict");
const {test} = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const atlas = require("../demos/perceptual_atlas.js");

test("normalization bounds values and rejects unknown formats", () => {
  const s = atlas.normalizeScene({seed:-5,mode:"unknown",light:9,
    detail:Infinity,flow:-1,pitch:"0.7",coupling:null});
  assert.equal(s.seed,0);
  assert.equal(s.mode,"spiral");
  assert.equal(s.light,1);
  assert.equal(s.detail,atlas.DEFAULT.detail);
  assert.equal(s.flow,0);
  assert.equal(s.pitch,0.7);
  assert.equal(s.coupling,atlas.DEFAULT.coupling);
  assert.throws(()=>atlas.normalizeSketch({format:"other"}),/format/);
});

test("generative field is deterministic, bounded, and lighting independent", () => {
  const s = atlas.normalizeScene({mode:"fold",seed:48});
  const a = atlas.field(0.34,-0.42,3.2,s);
  const b = atlas.field(0.34,-0.42,3.2,{...s,light:1});
  assert.deepEqual(a,b);
  assert.ok(a.intensity >= 0 && a.intensity <= 1);
  assert.ok(a.hue >= 0 && a.hue <= 1);
  assert.deepEqual(atlas.rgb(0.34,-0.42,3.2,s),atlas.rgb(0.34,-0.42,3.2,s));
  assert.notDeepEqual(atlas.rgb(0.34,-0.42,3.2,s),
    atlas.rgb(0.34,-0.42,3.2,{...s,light:1}));
});

test("geometry modes and pitch coupling actually affect patterns", () => {
  const s = atlas.normalizeScene({coupling:1,pitch:0.2});
  const positions = [[0.21,0.13],[0.7,-0.3],[-0.3,0.2]];
  const signature = scene => positions.map(([x,y])=>atlas.field(x,y,2.5,scene).intensity);
  assert.notDeepEqual(signature({...s,mode:"spiral"}),signature({...s,mode:"lattice"}));
  assert.notDeepEqual(signature({...s,pitch:0.1}),signature({...s,pitch:0.9}));
  assert.deepEqual(signature({...s,coupling:0,pitch:0.1}),
    signature({...s,coupling:0,pitch:0.9}));
});

test("rendered channels remain byte-range for all environments", () => {
  for (const mode of atlas.MODES) {
    for (const light of [0,0.2,0.5,0.8,1]) {
      const pixels = atlas.rgb(-0.8,0.71,5,{...atlas.DEFAULT,mode,light});
      assert.equal(pixels.length,3);
      for (const value of pixels) assert.ok(Number.isInteger(value) && value>=0 && value<=255);
    }
  }
});

test("time-indexed records replay faithfully after JSON roundtrip", () => {
  const initial = atlas.normalizeScene({seed:17,mode:"fold"});
  const next = atlas.normalizeScene({...initial,mode:"chamber",light:0.9});
  const sketch = atlas.normalizeSketch({
    initial,events:[{t:3,scene:next},{t:1,scene:initial}],duration:4
  });
  assert.equal(sketch.format,atlas.FORMAT);
  assert.deepEqual(atlas.sceneAt(sketch,0),initial);
  assert.deepEqual(atlas.sceneAt(sketch,2),initial);
  assert.deepEqual(atlas.sceneAt(sketch,3),next);
  assert.deepEqual(atlas.sceneAt(sketch,10),next);
  const copy = atlas.normalizeSketch(JSON.parse(JSON.stringify(sketch)));
  for (const t of [0,1,2.9,3,4]) assert.deepEqual(atlas.sceneAt(copy,t),atlas.sceneAt(sketch,t));
});

test("portable scene file and browser entrypoint are self-contained", () => {
  const html = fs.readFileSync(path.join(__dirname,"../demos/perceptual_atlas.html"),"utf8");
  for (const token of ['src="perceptual_atlas.js"','id="light"','id="record"',
    'id="save"','id="load"','id="replay"','requestAnimationFrame']) {
    assert.ok(html.includes(token),token);
  }
  assert.ok(!/https?:\/\/[^ \n"'<>]+/.test(html),"no external URLs or third-party dependencies");
  assert.ok(atlas.toneHz(atlas.DEFAULT)>0);
});
