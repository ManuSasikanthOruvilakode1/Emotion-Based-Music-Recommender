"use strict";

const EMOJI = {
  angry: "😠", contempt: "😒", disgust: "🤢", fear: "😨",
  happy: "😄", neutral: "😐", sad: "😢", surprise: "😮",
};
const FPS = 6;
const SEND_WIDTH = 480;

const $ = (sel) => document.querySelector(sel);
const state = { labels: [], strategy: "match", minConfidence: 0.4, last: null, reading: null, ws: null, stream: null };

// -- rendering -------------------------------------------------------------------------------
function buildBars() {
  $("#bars").innerHTML = state.labels
    .map(
      (l) => `<div class="bar" role="listitem" data-label="${l}">
        <span class="bar-name">${l}</span>
        <span class="bar-track"><span class="bar-fill"></span></span>
        <span class="bar-value">–</span></div>`
    )
    .join("");
}

function renderReading({ emotion, confidence, probs, forS, pending }) {
  const ok = emotion && confidence != null;
  state.reading = ok ? { emotion, confidence } : null;
  const unsure = ok && confidence < state.minConfidence;
  $("#emoji").textContent = ok ? EMOJI[emotion] ?? "·" : "·";
  $("#emotion").textContent = !ok ? "No face in view" : unsure ? `Not sure (maybe ${emotion})` : emotion;
  const meta = [];
  if (ok) meta.push(`${Math.round(confidence * 100)}% confidence`);
  if (forS != null) meta.push(`steady for ${forS.toFixed(0)} s`);
  if (pending) meta.push(`changing to ${pending}…`);
  $("#emotion-meta").textContent = meta.join(" · ");
  for (const row of document.querySelectorAll(".bar")) {
    const p = probs ? probs[row.dataset.label] ?? 0 : 0;
    row.querySelector(".bar-fill").style.width = `${(p * 100).toFixed(1)}%`;
    row.querySelector(".bar-value").textContent = probs ? `${Math.round(p * 100)}%` : "–";
    row.classList.toggle("top", ok && row.dataset.label === emotion);
  }
}

function renderReco(r) {
  state.last = r;
  const main = $("#reco-main");
  main.hidden = !r;
  if (!r) return;
  main.href = r.url;
  $("#reco-query").textContent = r.query;
  const notes = [];
  if (r.uncertain) notes.push("No confident reading, so here is something neutral.");
  else if (r.strategy !== r.applied) notes.push(`Nothing to lift for “${r.emotion}”, so this matches it.`);
  $("#reco-note").textContent = notes.join(" ");
  $("#reco-alt").innerHTML = r.alternatives
    .map((a) => `<li><a href="${a.url}" target="_blank" rel="noopener noreferrer">${escapeHtml(a.query)}</a></li>`)
    .join("");
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function drawBoxes(canvas, media, frame, faces) {
  const w = media.clientWidth, h = media.clientHeight;
  canvas.width = w * devicePixelRatio;
  canvas.height = h * devicePixelRatio;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  ctx.clearRect(0, 0, w, h);
  // media is shown with object-fit: contain; map frame pixels to displayed pixels
  const s = Math.min(w / frame.width, h / frame.height);
  const ox = (w - frame.width * s) / 2, oy = (h - frame.height * s) / 2;
  const accent = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
  faces.forEach((f, i) => {
    const [x, y, bw, bh] = f.box;
    ctx.lineWidth = i === 0 ? 3 : 1.5;
    ctx.strokeStyle = i === 0 ? accent : "rgba(255,255,255,.7)";
    ctx.strokeRect(ox + x * s, oy + y * s, bw * s, bh * s);
  });
}

// -- camera ----------------------------------------------------------------------------------
const video = $("#video");
const grab = document.createElement("canvas");

async function startCamera() {
  const btn = $("#camera-toggle");
  btn.disabled = true;
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
  } catch (e) {
    $("#status").textContent = `Camera unavailable (${e.name}). Try the Photo tab.`;
    btn.disabled = false;
    return;
  }
  video.srcObject = state.stream;
  await video.play();
  $("#camera-empty").hidden = true;
  btn.textContent = "Stop camera";
  btn.disabled = false;
  connect();
}

function stopCamera() {
  state.ws?.close();
  state.ws = null;
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
  video.srcObject = null;
  $("#camera-empty").hidden = false;
  $("#camera-toggle").textContent = "Start camera";
  $("#status").textContent = "";
  $("#overlay").getContext("2d").clearRect(0, 0, 1e4, 1e4);
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/stream`);
  ws.binaryType = "arraybuffer";
  state.ws = ws;
  let sentAt = 0;

  const sendFrame = () => {
    if (ws.readyState !== WebSocket.OPEN || !video.videoWidth) return;
    const scale = Math.min(1, SEND_WIDTH / video.videoWidth);
    grab.width = Math.round(video.videoWidth * scale);
    grab.height = Math.round(video.videoHeight * scale);
    grab.getContext("2d").drawImage(video, 0, 0, grab.width, grab.height);
    grab.toBlob((blob) => {
      if (blob && ws.readyState === WebSocket.OPEN) {
        sentAt = performance.now();
        ws.send(blob);
      }
    }, "image/jpeg", 0.8);
  };

  ws.onopen = () => {
    ws.send(JSON.stringify({ strategy: state.strategy }));
    sendFrame();
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    const rtt = performance.now() - sentAt;
    if (msg.error) {
      $("#status").textContent = msg.error;
    } else {
      const s = msg.smoothed;
      renderReading({ emotion: s.emotion, confidence: s.confidence, probs: s.probs, forS: s.for_s, pending: s.pending });
      renderReco(msg.recommendation);
      drawBoxes($("#overlay"), video, msg.frame, msg.faces);
      $("#status").textContent = `${msg.faces.length} face${msg.faces.length === 1 ? "" : "s"} · ${Math.round(rtt)} ms`;
    }
    setTimeout(sendFrame, Math.max(0, 1000 / FPS - rtt));
  };
  ws.onclose = () => {
    if (state.ws === ws && state.stream) $("#status").textContent = "Connection lost. Stop and start the camera to retry.";
  };
}

$("#camera-toggle").addEventListener("click", () => (state.stream ? stopCamera() : startCamera()));

// -- photo -----------------------------------------------------------------------------------
$("#file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const img = $("#photo");
  if (img.src) URL.revokeObjectURL(img.src);
  img.src = URL.createObjectURL(file);
  $("#photo-empty").hidden = true;
  $("#photo-status").textContent = "Analysing…";
  const body = new FormData();
  body.append("file", file);
  try {
    const r = await fetch(`/api/analyze?strategy=${state.strategy}`, { method: "POST", body });
    const msg = await r.json();
    if (!r.ok) throw new Error(msg.detail ?? r.statusText);
    const main = msg.faces[0];
    renderReading(main ? { emotion: main.emotion, confidence: main.confidence, probs: main.probs } : {});
    renderReco(msg.recommendation);
    await img.decode().catch(() => {});
    drawBoxes($("#photo-overlay"), img, msg.frame, msg.faces);
    $("#photo-status").textContent = main ? `${msg.faces.length} face${msg.faces.length === 1 ? "" : "s"} found` : "No face found";
  } catch (err) {
    $("#photo-status").textContent = `Could not analyse: ${err.message}`;
  }
});

// -- tabs and strategy -----------------------------------------------------------------------
for (const tab of document.querySelectorAll("[role=tab]")) {
  tab.addEventListener("click", () => {
    for (const t of document.querySelectorAll("[role=tab]")) {
      const on = t === tab;
      t.setAttribute("aria-selected", on);
      $(`#${t.getAttribute("aria-controls")}`).hidden = !on;
    }
    if (tab.id === "tab-photo" && state.stream) stopCamera();
  });
}

function setStrategy(s) {
  state.strategy = s;
  for (const b of document.querySelectorAll("[data-strategy]")) b.setAttribute("aria-checked", b.dataset.strategy === s);
  if (state.ws?.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({ strategy: s }));
  } else if (state.last) {
    const q = new URLSearchParams({ strategy: s });
    if (state.reading) {
      q.set("emotion", state.reading.emotion);
      q.set("confidence", state.reading.confidence);
    }
    fetch(`/api/recommend?${q}`).then((r) => r.json()).then(renderReco);
  }
}
for (const b of document.querySelectorAll("[data-strategy]")) b.addEventListener("click", () => setStrategy(b.dataset.strategy));

fetch("/api/meta")
  .then((r) => r.json())
  .then((m) => {
    state.labels = m.labels;
    state.minConfidence = m.min_confidence;
    buildBars();
    setStrategy(m.default_strategy);
  });
