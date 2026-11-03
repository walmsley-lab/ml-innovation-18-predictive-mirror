"""Portable HTML/PNG storyboard for verified, open-loop 3D forecasts."""
import base64
import json
import struct
import zlib
import numpy as np


def png_data_url(image):
    rgb=np.clip(np.asarray(image)*255,0,255).astype(np.uint8)
    if rgb.ndim!=3 or rgb.shape[-1]!=3:
        raise ValueError("Expected RGB image")
    height,width,_=rgb.shape
    def chunk(tag,data):
        return (struct.pack(">I",len(data))+tag+data+
                struct.pack(">I",zlib.crc32(tag+data)&0xffffffff))
    scanlines=b"".join(b"\x00"+rgb[i].tobytes() for i in range(height))
    header=struct.pack(">IIBBBBB",width,height,8,2,0,0,0)
    encoded=(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",header)+
             chunk(b"IDAT",zlib.compress(scanlines,6))+chunk(b"IEND",b""))
    return "data:image/png;base64,"+base64.b64encode(encoded).decode("ascii")


def html(last,predicted,actual,persistence,title="Predictive 3D mirror"):
    if (predicted.shape!=actual.shape or persistence.shape!=actual.shape
            or predicted.ndim!=4 or last.shape!=predicted.shape[1:]):
        raise ValueError("Incompatible storyboard frames")
    pictures=[]
    for forecast,truth,hold in zip(predicted,actual,persistence):
        error=np.clip(np.abs(forecast-truth)*4,0,1)
        pictures.append([png_data_url(last),png_data_url(forecast),
                         png_data_url(truth),png_data_url(hold),
                         png_data_url(error)])
    data=json.dumps(pictures)
    headline=json.dumps(title)
    # Inline JS is a local file navigator only: no external libraries,
    # server calls, or hidden browser-accessible future prediction inputs.
    return """<!doctype html><html lang="en"><meta charset="utf-8">
<title>Predictive 3D mirror</title><style>
body{font:15px system-ui;max-width:1000px;margin:2em auto;padding:0 1em}
section{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0}
img{width:170px;height:170px;image-rendering:pixelated;border:1px solid #999}
figcaption{margin:.4em 0}input{width:85%}</style>
<h1 id="title"></h1><p>Open-loop forecast committed before future
observation; future frames below are evaluation-only ground truth.</p>
<label>Future lead: <strong id="count"></strong></label>
<input type="range" id="lead" min="1" max="N" value="1">
<section id="pictures"></section>
<script>
const frames = DATA;
const names = ["Last observed","Predicted","Later observed",
               "Persistence control","Absolute RGB error ×4"];
document.getElementById("title").textContent = TITLE;
const slider = document.getElementById("lead");
slider.max=frames.length;
function show(){
 const i=Number(slider.value)-1;
 document.getElementById("count").textContent=String(i+1);
 const container=document.getElementById("pictures");
 container.replaceChildren();
 frames[i].forEach((source,j)=>{
  const figure=document.createElement("figure");
  const pic=document.createElement("img"); pic.src=source;
  const caption=document.createElement("figcaption");
  caption.textContent=names[j];figure.append(pic,caption);
  container.append(figure);
 });
}
slider.addEventListener("input",show);show();
</script></html>""".replace("DATA",data).replace("TITLE",headline).replace('max="N"',f'max="{len(pictures)}"')
