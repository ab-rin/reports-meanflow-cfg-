"use strict";

(() => {
  const data = window.PROJECT_DATA;
  if (!data) return;
  const $ = (id) => document.getElementById(id);
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const svgNS = "http://www.w3.org/2000/svg";
  function svg(tag, attrs = {}, text) {
    const element = document.createElementNS(svgNS, tag);
    Object.entries(attrs).forEach(([key, value]) => element.setAttribute(key, value));
    if (text !== undefined) element.textContent = text;
    return element;
  }
  function textElement(tag, text, className) {
    const element = document.createElement(tag);
    element.textContent = text;
    if (className) element.className = className;
    return element;
  }
  const species = [...new Set(data.gallery.samples.map((sample) => sample.species))];
  const seeds = [...new Set(data.gallery.samples.map((sample) => sample.seed))].sort((a, b) => a - b);
  let selectedSpecies = species[0];
  let playTimer = null;
  let frame = 4;
  let imageRequest = 0;
  let playTarget = null;
  const cards = Array.from({ length: 5 }, () => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "seed-card";
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 64;
    canvas.setAttribute("aria-hidden", "true");
    const label = textElement("span", "");
    button.append(canvas, label);
    $("seed-gallery").append(button);
    return { button, canvas, label, image: null, seed: 42 };
  });
  $("seed").max = seeds.length - 1;
  $("seed").value = seeds.indexOf(42);
  function setFrame(index) {
    frame = index;
    cards.forEach((card, i) => {
      if (!card.image) return;
      const cardFrame = playTarget === null || playTarget === i ? frame : 4;
      card.canvas.getContext("2d").drawImage(card.image, 64 * cardFrame, 0, 64, 64, 0, 0, 64, 64);
      card.button.setAttribute("aria-label", `Replay ${selectedSpecies}, seed ${card.seed}, ${cardFrame ? "after call " + cardFrame : "initial noise"}`);
    });
    [...$("generation-progress").children].forEach((mark, i) => mark.classList.toggle("filled", i < frame));
    $("generation-status").textContent = frame === 4 ? "" : frame ? `Call ${frame} of 4` : "Starting from noise…";
  }
  function stopPlayback() {
    clearInterval(playTimer);
    playTimer = null;
    $("play").textContent = "↺ Replay all five";
  }
  async function loadTrajectory() {
    stopPlayback();
    const request = ++imageRequest;
    const first = +$("seed").value;
    $("selected-species").textContent = selectedSpecies;
    $("seed-value").textContent = seeds[first];
    $("seed").setAttribute("aria-valuetext", `Five samples starting at seed ${seeds[first]}`);
    document.querySelectorAll("#sample-shelf button").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.species === selectedSpecies)));
    $("play").disabled = true;
    $("generation-status").textContent = "Loading…";
    const samples = cards.map((card, index) => {
      card.seed = seeds[(first + index) % seeds.length];
      card.image = null;
      card.button.disabled = true;
      card.button.setAttribute("aria-label", `Loading ${selectedSpecies}, seed ${card.seed}`);
      card.label.textContent = `Seed ${card.seed}`;
      card.canvas.getContext("2d").clearRect(0, 0, 64, 64);
      return data.gallery.samples.find((sample) => sample.species === selectedSpecies && sample.seed === card.seed);
    });
    try {
      const images = await Promise.all(samples.map((sample) => new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = () => resolve(image);
        image.onerror = reject;
        image.src = `assets/gallery/${sample.trajectory}`;
      })));
      if (request !== imageRequest) return;
      cards.forEach((card, index) => { card.image = images[index]; card.button.disabled = false; });
      playTarget = null;
      setFrame(4);
      $("play").disabled = false;
    } catch {
      if (request === imageRequest) $("generation-status").textContent = "Images unavailable. Try another set of seeds.";
    }
  }
  function startPlayback(target = null) {
    stopPlayback();
    playTarget = target;
    setFrame(0);
    $("play").textContent = "Ⅱ Pause";
    playTimer = setInterval(() => {
      setFrame(frame + 1);
      if (frame === 4) stopPlayback();
    }, 700);
  }
  cards.forEach((card, index) => card.button.addEventListener("click", () => startPlayback(index)));
  $("seed").addEventListener("input", loadTrajectory);
  $("another-seed").addEventListener("click", () => {
    $("seed").value = (+$("seed").value + cards.length) % seeds.length;
    loadTrajectory();
  });
  $("play").addEventListener("click", () => {
    if (playTimer) stopPlayback(); else startPlayback();
  });
  $("pixel-toggle").addEventListener("click", (event) => {
    const active = document.body.classList.toggle("pixelated");
    event.target.setAttribute("aria-pressed", String(active));
    event.target.textContent = active ? "Pixels on" : "Pixels off";
  });

  // Decorative transitions between recorded outputs; no extra model states.
  const specimenCanvas = $("type-canvas");
  const specimenContext = specimenCanvas.getContext("2d");
  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = sourceCanvas.height = 64;
  const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
  const specimenSamples = data.gallery.samples.filter((sample) => sample.seed === 42);
  const specimenLayers = new Map();
  let specimenIndex = 0;
  let specimenLayer = null;
  let specimenTimer = null;
  let specimenAnimation = null;
  let specimenRequest = 0;
  let specimenPaused = false;
  let specimenVisible = true;
  const specimenFont = document.fonts.load("8px Plex");
  function specimenImage(sample) {
    if (!specimenLayers.has(sample.species)) {
      specimenLayers.set(sample.species, new Promise((resolve, reject) => {
        const image = new Image();
        image.onload = async () => {
          await specimenFont;
          sourceContext.clearRect(0, 0, 64, 64);
          sourceContext.drawImage(image, 0, 0);
          const pixels = sourceContext.getImageData(0, 0, 64, 64).data;
          const layer = document.createElement("canvas");
          layer.width = layer.height = 480;
          const context = layer.getContext("2d");
          const alphabet = " .·:+*ox#@";
          context.font = "8px Plex, monospace";
          context.textAlign = "center";
          for (let y = 0; y < 64; y++) {
            for (let x = 0; x < 64; x++) {
              const i = (y * 64 + x) * 4;
              const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
              const brightness = (r + g + b) / 3;
              if (brightness > 244) continue;
              const glyph = alphabet[Math.min(9, Math.floor((255 - brightness) / 255 * 9) + 1)];
              context.fillStyle = `rgb(${Math.round(r * .54)},${Math.round(g * .54)},${Math.round(b * .54)})`;
              context.fillText(glyph, x * 7.5 + 3.75, y * 7.5 + 7);
            }
          }
          resolve(layer);
        };
        image.onerror = reject;
        image.src = "assets/gallery/" + sample.image;
      }));
    }
    return specimenLayers.get(sample.species);
  }
  function drawSpecimen(layer = specimenLayer, opacity = 1, offset = 0) {
    specimenContext.clearRect(0, 0, 480, 480);
    if (!layer) return;
    specimenContext.globalAlpha = opacity;
    specimenContext.drawImage(layer, 0, offset);
    specimenContext.globalAlpha = 1;
  }
  function stopSpecimen() {
    clearTimeout(specimenTimer);
    cancelAnimationFrame(specimenAnimation);
    specimenTimer = specimenAnimation = null;
    drawSpecimen();
  }
  function scheduleSpecimen() {
    clearTimeout(specimenTimer);
    if (specimenPaused || reducedMotion.matches || document.hidden || !specimenVisible) return;
    specimenTimer = setTimeout(() => showSpecimen((specimenIndex + 1) % specimenSamples.length), 3500);
  }
  function updateSpecimenControl() {
    const button = $("specimen-toggle");
    const label = reducedMotion.matches ? "Show another Pokémon" : specimenPaused ? "Resume changing Pokémon" : "Pause changing Pokémon";
    button.setAttribute("aria-label", label);
    button.title = label;
    button.setAttribute("aria-pressed", String(specimenPaused));
    button.classList.toggle("paused", specimenPaused);
    button.querySelector("span").textContent = specimenPaused || reducedMotion.matches ? "▶" : "Ⅱ";
  }
  async function showSpecimen(index, animate = true) {
    const request = ++specimenRequest;
    stopSpecimen();
    try {
      const next = await specimenImage(specimenSamples[index]);
      if (request !== specimenRequest) return;
      const previous = specimenLayer;
      specimenIndex = index;
      specimenLayer = next;
      specimenCanvas.setAttribute("aria-label", "Typographic rendering of generated " + specimenSamples[index].species);
      document.querySelectorAll("#hero-pokemon button").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.species === specimenSamples[index].species)));
      if (!animate || !previous || reducedMotion.matches || document.hidden || !specimenVisible) {
        drawSpecimen(); scheduleSpecimen(); return;
      }
      const started = performance.now();
      const ease = (value) => value * value * (3 - 2 * value);
      function dissolve(time) {
        const progress = Math.min(1, (time - started) / 1200);
        if (progress < .5) drawSpecimen(previous, 1 - ease(progress * 2), -4 * progress);
        else drawSpecimen(next, ease((progress - .5) * 2), 4 * (1 - progress));
        if (progress < 1) specimenAnimation = requestAnimationFrame(dissolve);
        else { specimenAnimation = null; drawSpecimen(); scheduleSpecimen(); }
      }
      specimenAnimation = requestAnimationFrame(dissolve);
    } catch {
      if (request === specimenRequest) scheduleSpecimen();
    }
  }
  $("specimen-toggle").addEventListener("click", () => {
    if (reducedMotion.matches) { showSpecimen((specimenIndex + 1) % specimenSamples.length, false); return; }
    specimenPaused = !specimenPaused;
    updateSpecimenControl();
    if (specimenPaused) stopSpecimen(); else scheduleSpecimen();
  });
  ["Mew", "Gengar", "Dragonite"].forEach((name) => {
    const index = specimenSamples.findIndex((sample) => sample.species === name);
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.species = name;
    button.setAttribute("aria-label", "Show " + name + " in the artwork");
    button.title = name;
    button.setAttribute("aria-pressed", "false");
    const image = new Image();
    image.src = "assets/gallery/" + specimenSamples[index].image;
    image.alt = ""; image.width = image.height = 64;
    button.append(image);
    button.addEventListener("click", () => showSpecimen(index));
    $("hero-pokemon").append(button);
  });
  const mewAudio = $("mew-audio");
  mewAudio.volume = .35;
  $("mew-cry").addEventListener("click", async () => {
    showSpecimen(specimenSamples.findIndex((sample) => sample.species === "Mew"));
    mewAudio.currentTime = 0;
    try {
      await mewAudio.play();
      $("mew-cry").classList.add("playing");
      $("audio-status").textContent = "Playing Mew’s cry";
    } catch {
      $("audio-status").textContent = "Mew’s cry could not be played in this browser.";
    }
  });
  mewAudio.addEventListener("ended", () => { $("mew-cry").classList.remove("playing"); $("audio-status").textContent = ""; });
  new IntersectionObserver(([entry]) => {
    specimenVisible = entry.isIntersecting;
    if (specimenVisible) scheduleSpecimen(); else stopSpecimen();
  }, { threshold: .1 }).observe($("top"));
  reducedMotion.addEventListener("change", () => {
    updateSpecimenControl(); stopSpecimen(); scheduleSpecimen();
  });
  updateSpecimenControl();
  specimenSamples.forEach((sample) => {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-label", "Explore " + sample.species);
    button.dataset.species = sample.species;
    const image = new Image();
    image.src = "assets/gallery/" + sample.image;
    image.alt = "";
    image.width = image.height = 64;
    button.append(image, textElement("span", sample.species.toLowerCase()));
    button.addEventListener("click", () => {
      selectedSpecies = sample.species;
      loadTrajectory();
    });
    $("sample-shelf").append(button);
  });
  loadTrajectory();
  showSpecimen(0, false);

  // The original opening-only trail: samples appear in place, then fade.
  const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)");
  let trailIndex = 0;
  let trailPoint = null;
  function stopTrail() {
    trailPoint = null;
    $("sample-trail").replaceChildren();
  }
  $("top").addEventListener("pointermove", (event) => {
    if (reducedMotion.matches || !finePointer.matches || event.pointerType !== "mouse" ||
        event.target.closest("a,button,canvas,.intro-note,.intro-title")) return;
    const bounds = $("top").getBoundingClientRect();
    const point = [event.clientX - bounds.left, event.clientY - bounds.top];
    if (trailPoint && Math.hypot(point[0] - trailPoint[0], point[1] - trailPoint[1]) < 95) return;
    trailPoint = point;
    const sample = specimenSamples[trailIndex % specimenSamples.length];
    const image = new Image();
    image.src = "assets/gallery/" + sample.image;
    image.className = "trail-sample";
    image.alt = "";
    image.style.left = (point[0] - 24) + "px";
    image.style.top = (point[1] - 24) + "px";
    image.style.rotate = ((trailIndex % 5) * 7 - 14) + "deg";
    trailIndex++;
    $("sample-trail").append(image);
    if ($("sample-trail").children.length > 10) $("sample-trail").firstElementChild.remove();
    image.addEventListener("animationend", () => image.remove(), { once: true });
  });
  $("top").addEventListener("pointerleave", () => { trailPoint = null; });
  window.addEventListener("blur", stopTrail);
  finePointer.addEventListener("change", stopTrail);
  reducedMotion.addEventListener("change", stopTrail);

  const trainingStages = [...document.querySelectorAll("[data-training-stage]")];
  const stageDescriptions = [
    "Mix a training image with Gaussian noise at time t. Keep the image’s class label as the condition.",
    "The class-conditioned U-Net predicts average velocity over the interval from t to r.",
    "Combine guided velocity with a Jacobian-vector product (JVP). Detach this target so gradients do not flow through it.",
    "Compare prediction and target with the adaptively weighted loss. Adam updates the network; an exponential moving average (EMA) keeps the sampling weights."
  ];
  let trainingStage = 0;
  let diagramTimer = null;
  function showTrainingStage(index) {
    trainingStage = index;
    trainingStages.forEach((button, i) => button.setAttribute("aria-pressed", String(i === index)));
    $("training-explanation").textContent = stageDescriptions[index];
  }
  function stopDiagram() {
    clearInterval(diagramTimer);
    diagramTimer = null;
    $("motion-toggle").textContent = "▶ Play loop";
    $("motion-toggle").setAttribute("aria-pressed", "false");
  }
  trainingStages.forEach((button, index) => button.addEventListener("click", () => { stopDiagram(); showTrainingStage(index); }));
  $("motion-toggle").addEventListener("click", () => {
    if (diagramTimer) { stopDiagram(); return; }
    showTrainingStage(0);
    $("motion-toggle").textContent = "Ⅱ Pause loop";
    $("motion-toggle").setAttribute("aria-pressed", "true");
    diagramTimer = setInterval(() => showTrainingStage((trainingStage + 1) % trainingStages.length), 1800);
  });
  reducedMotion.addEventListener("change", () => { if (reducedMotion.matches) stopDiagram(); });

  let chartZoom = false;
  let chartTimer = null;
  let chartIndex = data.history.length - 1;
  $("training-step").max = chartIndex;
  $("training-step").value = chartIndex;
  function drawFID() {
    const chart = $("fid-chart");
    chart.replaceChildren(svg("title", {}, "Validation FID during training"),
      svg("desc", {}, "Measured uniform-four-call validation scores. Context lines show the course target and real-data reference, not lower bounds."));
    const width = Math.max(280, chart.clientWidth);
    chart.setAttribute("viewBox", "0 0 " + width + " 360");
    const left = 45, right = width - 15, top = 25, bottom = 300;
    const xmax = Math.max(110000, data.history.at(-1)[0]);
    const xmin = chartZoom ? 60000 : 0;
    const ymin = chartZoom ? 19 : 15, ymax = chartZoom ? 28 : 135;
    const x = (step) => left + (step - xmin) / (xmax - xmin) * (right - left);
    const y = (fid) => bottom - (fid - ymin) / (ymax - ymin) * (bottom - top);
    const defs = svg("defs");
    const clip = svg("clipPath", { id: "fid-clip" });
    clip.append(svg("rect", { x: left, y: top, width: right - left, height: bottom - top }));
    defs.append(clip); chart.append(defs);
    const yTicks = chartZoom ? [20, 22, 24, 26, 28] : [20, 40, 60, 80, 100, 120];
    yTicks.forEach((value) => {
      chart.append(svg("line", { x1: left, x2: right, y1: y(value), y2: y(value), class: "chart-grid" }),
        svg("text", { x: left - 15, y: y(value) + 4, "text-anchor": "end", class: "chart-axis" }, value));
    });
    [60000, 80000, 100000, 120000].concat(chartZoom ? [] : [0, 20000, 40000]).sort((a, b) => a - b).filter((v) => v <= xmax).forEach((step) => {
      chart.append(svg("text", { x: x(step), y: 327, "text-anchor": "middle", class: "chart-axis" }, `${step / 1000}k`));
    });
    chart.append(svg("text", { x: right, y: 352, "text-anchor": "end", class: "chart-axis" }, "optimizer updates"),
      svg("text", { x: left, y: 13, "text-anchor": "start", class: "chart-axis" }, "FID ↓"));
    [[22, "#8799a5", "7 5"], [20.82, "#b4b285", "2 5"]].forEach(([value, color, dash]) => chart.append(svg("line", {
      x1: left, x2: right, y1: y(value), y2: y(value), stroke: color, "stroke-dasharray": dash, "stroke-width": 1.2
    })));
    const visible = data.history.slice(0, chartIndex + 1);
    const path = visible.map(([step, score], i) => `${i ? "L" : "M"}${x(step).toFixed(2)},${y(score).toFixed(2)}`).join(" ");
    chart.append(svg("path", { d: path, class: "fid-path", "clip-path": "url(#fid-clip)" }));
    const point = data.history[chartIndex];
    if (point[0] >= xmin && point[1] <= ymax && point[1] >= ymin) {
      chart.append(svg("line", { x1: x(point[0]), x2: x(point[0]), y1: y(point[1]), y2: bottom, stroke: "#47675b55", "stroke-dasharray": "3 4" }),
        svg("circle", { cx: x(point[0]), cy: y(point[1]), r: 5, fill: "#47675b", stroke: "#fafbf9", "stroke-width": 2 }));
    }
    const snapshot = [1000, 8000, 32000, 64000, 101000, 122000].filter((step) => step <= point[0]).at(-1) || 1000;
    const snapshotPath = `assets/study/journey-${snapshot}.png`;
    if ($("training-image").getAttribute("src") !== snapshotPath) {
      $("training-image").src = snapshotPath;
      $("training-image").alt = `Pikachu at ${snapshot.toLocaleString()} training updates, seed 42`;
      $("training-image-caption").textContent = `Saved at ${snapshot / 1000}k updates`;
    }
    $("training-readout").textContent = `${point[0].toLocaleString()} updates · ${point[1].toFixed(2)} FID`;
    $("training-step").value = chartIndex;
    $("training-step").setAttribute("aria-valuetext", $("training-readout").textContent);
  }
  function stopChart() {
    clearInterval(chartTimer);
    chartTimer = null;
    $("replay-chart").textContent = "↻ Replay training";
  }
  $("training-step").addEventListener("input", (event) => { stopChart(); chartIndex = +event.target.value; drawFID(); });
  $("zoom-chart").addEventListener("click", (event) => {
    chartZoom = !chartZoom;
    event.target.setAttribute("aria-pressed", String(chartZoom));
    event.target.textContent = chartZoom ? "Show the full run" : "Zoom near 22";
    $("training-step").min = chartZoom ? data.history.findIndex((p) => p[0] >= 60000) : 0;
    if (chartZoom && data.history[chartIndex][0] < 60000) chartIndex = data.history.length - 1;
    drawFID();
  });
  $("replay-chart").addEventListener("click", () => {
    if (chartTimer) { stopChart(); return; }
    chartIndex = chartZoom ? data.history.findIndex((p) => p[0] >= 60000) : 0;
    drawFID();
    $("replay-chart").textContent = "Ⅱ Pause replay";
    chartTimer = setInterval(() => {
      chartIndex += 1; drawFID();
      if (chartIndex === data.history.length - 1) stopChart();
    }, 35);
  });
  drawFID();

  const schedules = { uniform: [1, .75, .5, .25, 0], rho2: [1, .5625, .25, .0625, 0], tuned: [1, .6, .3, .018, 0] };
  for (let i = 0; i < 4; i++) {
    const block = document.createElement("div");
    block.append(textElement("span", `call ${i + 1}`));
    $("schedule-bars").append(block);
  }
  function setSchedule(name) {
    const times = schedules[name];
    [...$("schedule-bars").children].forEach((block, i) => {
      const interval = times[i] - times[i + 1];
      block.style.flexGrow = interval;
      block.style.flexBasis = "0";
      block.title = `Call ${i + 1}: ${times[i]} → ${times[i + 1]}, interval ${interval.toFixed(3)}`;
    });
    $("schedule-times").textContent = times.join("  →  ");
    document.querySelectorAll("[data-schedule]").forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.schedule === name)));
  }
  document.querySelectorAll("[data-schedule]").forEach((button) => button.addEventListener("click", () => setSchedule(button.dataset.schedule)));
  setSchedule("tuned");
  function drawPaired() {
    const pairChart = $("paired-chart");
    const width = Math.max(280, pairChart.clientWidth);
    pairChart.setAttribute("viewBox", `0 0 ${width} 290`);
    pairChart.replaceChildren();
    const px = (value) => 70 + (value - 19) / 1.1 * (width - 95);
    [19, 19.25, 19.5, 19.75, 20].forEach((value) => {
      pairChart.append(svg("line", { x1: px(value), x2: px(value), y1: 40, y2: 228, stroke: "#e1ded3" }),
        svg("text", { x: px(value), y: 253, "text-anchor": "middle", class: "chart-axis" }, value.toFixed(2)));
    });
    data.pairs.forEach((row, index) => {
      const y = 55 + index * 39;
      pairChart.append(svg("text", { x: 55, y: y + 4, "text-anchor": "end", class: "chart-axis" }, `run ${index + 1}`),
        svg("line", { x1: px(row.tuned), x2: px(row.baseline), y1: y, y2: y, stroke: "#afb5a9", "stroke-width": 2 }),
        svg("circle", { cx: px(row.baseline), cy: y, r: 5, fill: "#8799a5" }),
        svg("circle", { cx: px(row.tuned), cy: y, r: 5, fill: "#47675b" }));
    });
    pairChart.append(svg("text", { x: 70, y: 282, fill: "#47675b", "font-size": 12 }, "● tuned"),
      svg("text", { x: 160, y: 282, fill: "#8799a5", "font-size": 12 }, "● ρ = 2"));
  }
  drawPaired();

  if (data.study) {
    const panel = $("checkpoint-comparison");
    panel.replaceChildren();
    let pairIndex = 0;
    let revealed = false;
    const controls = textElement("div", "", "comparison-controls");
    const pairName = textElement("h3", "");
    const next = textElement("button", "Another pair ↗", "quiet-button");
    next.type = "button";
    controls.append(pairName, next);
    const stage = textElement("div", "", "comparison-stage");
    const images = [new Image(), new Image()];
    images.forEach((image, i) => { image.width = image.height = 64; image.className = `comparison-image image-${i}`; });
    const divider = textElement("div", "↔", "comparison-divider");
    divider.setAttribute("aria-hidden", "true");
    const wipe = document.createElement("input");
    wipe.type = "range"; wipe.min = 0; wipe.max = 100; wipe.value = 50;
    wipe.id = "comparison-wipe";
    wipe.setAttribute("aria-label", "Compare checkpoints: slide left for B, right for A");
    const tagA = textElement("span", "A", "comparison-tag tag-a");
    const tagB = textElement("span", "B", "comparison-tag tag-b");
    stage.append(images[1], images[0], tagA, tagB, divider, wipe);
    function positionWipe() {
      stage.style.setProperty("--split", wipe.value + "%");
      wipe.setAttribute("aria-valuetext", `${wipe.value}% checkpoint A, ${100 - wipe.value}% checkpoint B`);
    }
    wipe.addEventListener("input", positionWipe);
    const actions = textElement("div", "", "comparison-actions");
    const reveal = textElement("button", "Reveal the scores", "button");
    reveal.type = "button";
    const scores = textElement("div", "", "comparison-scores");
    scores.id = "comparison-scores";
    reveal.setAttribute("aria-controls", scores.id);
    const pairSeed = textElement("span", "", "pair-seed");
    actions.append(pairSeed, reveal);
    function showComparison() {
      const name = data.study.species[pairIndex % data.study.species.length];
      const seed = data.study.visual_seeds[Math.floor(pairIndex / data.study.species.length)];
      pairName.textContent = name;
      pairSeed.textContent = `Seed ${seed}`;
      data.study.checkpoints.forEach((checkpoint, i) => {
        images[i].src = `assets/study/compare-${checkpoint.step}-${name.toLowerCase()}-${seed}.png`;
        images[i].alt = `${name}, checkpoint ${i ? "B" : "A"}, seed ${seed}`;
      });
      wipe.value = 50; positionWipe();
    }
    function showScores() {
      scores.hidden = !revealed;
      reveal.textContent = revealed ? "Hide the scores" : "Reveal the scores";
      reveal.setAttribute("aria-expanded", String(revealed));
    }
    data.study.checkpoints.forEach((checkpoint, i) => {
      const score = checkpoint.remeasured_uniform_fid ?? checkpoint.historical_uniform_fid;
      const column = textElement("p", "");
      column.append(textElement("span", `${i ? "B" : "A"} · ${checkpoint.step / 1000}k updates`), textElement("strong", `${score.toFixed(2)} FID`));
      scores.append(column);
    });
    next.addEventListener("click", () => {
      pairIndex = (pairIndex + 1) % (data.study.species.length * data.study.visual_seeds.length);
      showComparison();
    });
    reveal.addEventListener("click", () => { revealed = !revealed; showScores(); });
    panel.append(controls, stage, actions, scores);
    showComparison(); showScores();
  }

  function drawLoss() {
    if (!data.loss) return;
    const panel = $("loss-panel");
    panel.replaceChildren();
    panel.className = "loss-chart chart-panel";
    panel.append(textElement("h3", "Training loss."));
    const width = Math.max(280, panel.clientWidth);
    const lossSvg = svg("svg", { viewBox: "0 0 " + width + " 250", role: "img", "aria-label": "Recorded adaptive training loss, summarized in 1,000-update bins" });
    const x = (step) => 45 + step / data.loss.at(-1)[0] * (width - 60);
    const values = data.loss.flatMap((row) => [row[2], row[3]]);
    const low = Math.floor(Math.min(...values)), high = Math.ceil(Math.max(...values));
    const y = (value) => 205 - (value - low) / (high - low) * 180;
    for (let value = low; value <= high; value += 2) {
      lossSvg.append(svg("line", { x1: 45, x2: width - 15, y1: y(value), y2: y(value), class: "chart-grid" }),
        svg("text", { x: 30, y: y(value) + 4, "text-anchor": "end", class: "chart-axis" }, value));
    }
    const band = data.loss.map((row, i) => `${i ? "L" : "M"}${x(row[0])},${y(row[3])}`).join(" ") +
      data.loss.slice().reverse().map((row) => `L${x(row[0])},${y(row[2])}`).join(" ") + "Z";
    const path = data.loss.map((row, i) => `${i ? "L" : "M"}${x(row[0])},${y(row[1])}`).join(" ");
    lossSvg.append(svg("path", { d: band, fill: "#c9d1c3", opacity: ".6" }),
      svg("path", { d: path, fill: "none", stroke: "#8799a5", "stroke-width": 2 }));
    [0, 40000, 80000, 120000].forEach((step) => lossSvg.append(svg("text", { x: x(step), y: 233, "text-anchor": "middle", class: "chart-axis" }, `${step / 1000}k`)));
    panel.append(lossSvg);
    const notes = document.createElement("details");
    notes.append(textElement("summary", "About the loss curve"), textElement("p", "Mean logged adaptive loss in 1,000-update bins. Shading shows the minimum and maximum, not a confidence interval. The objective changes during training; this is not an image-quality score. Duplicate steps retain their last occurrence."));
    const link = textElement("a", "Download the numeric loss history ↗", "caption");
    link.href = "assets/study/training_loss.csv"; notes.append(link); panel.append(notes);
  }
  drawLoss();
  $("copy-command").addEventListener("click", async (event) => {
    try {
      await navigator.clipboard.writeText($("sample-command").textContent);
      event.target.textContent = "Copied";
    } catch {
      const range = document.createRange();
      range.selectNodeContents($("sample-command"));
      const selection = window.getSelection();
      selection.removeAllRanges(); selection.addRange(range);
      event.target.textContent = "Selected — copy manually";
    }
    setTimeout(() => { event.target.textContent = "Copy command"; }, 2400);
  });
  window.addEventListener("resize", () => { drawFID(); drawLoss(); drawPaired(); });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { stopPlayback(); stopChart(); stopSpecimen(); stopTrail(); stopDiagram(); mewAudio.pause(); $("mew-cry").classList.remove("playing"); }
    else scheduleSpecimen();
  });
})();
