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
  $("species").replaceChildren(...species.map((name) => new Option(name, name)));
  let playTimer = null;
  let frame = 4;
  let imageRequest = 0;
  const frames = [];
  data.gallery.times.forEach((time, index) => {
    const figure = document.createElement("figure");
    figure.className = "state-card";
    const button = document.createElement("button");
    button.type = "button";
    button.className = "state-button";
    button.setAttribute("aria-label", `Inspect ${index ? "call " + index : "initial noise"}, time ${time}`);
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = 64;
    canvas.className = "state-image";
    canvas.setAttribute("aria-hidden", "true");
    button.append(canvas);
    button.addEventListener("click", () => { stopPlayback(); setFrame(index); });
    const caption = document.createElement("figcaption");
    caption.append(textElement("span", index ? `CALL ${index}` : "NOISE"), textElement("span", `t = ${time}`));
    figure.append(button, caption);
    $("trajectory").append(figure);
    frames.push({ figure, canvas, button });
  });
  function setFrame(index) {
    frame = index;
    frames.forEach(({ figure, button }, i) => {
      figure.classList.toggle("current", i === frame);
      button.setAttribute("aria-pressed", String(i === frame));
    });
    $("step").value = frame;
    $("step-label").textContent = `${frame ? "Call " + frame : "Noise"} · t = ${data.gallery.times[frame]}`;
    $("step").setAttribute("aria-valuetext", $("step-label").textContent);
  }
  function stopPlayback() {
    clearInterval(playTimer);
    playTimer = null;
    $("play").textContent = "▶ Play four calls";
  }
  function loadTrajectory() {
    stopPlayback();
    const request = ++imageRequest;
    const sample = data.gallery.samples.find((s) => s.species === $("species").value && s.seed === +$("seed").value);
    const image = new Image();
    $("play").disabled = true;
    image.onload = () => {
      if (request !== imageRequest) return;
      frames.forEach(({ canvas }, index) => canvas.getContext("2d").drawImage(image, 64 * index, 0, 64, 64, 0, 0, 64, 64));
      $("play").disabled = false;
      setFrame(4);
    };
    image.onerror = () => {
      if (request === imageRequest) $("step-label").textContent = "Image unavailable";
    };
    image.src = `assets/gallery/${sample.trajectory}`;
  }
  $("species").addEventListener("change", loadTrajectory);
  $("seed").addEventListener("change", loadTrajectory);
  $("step").addEventListener("input", (event) => { stopPlayback(); setFrame(+event.target.value); });
  $("play").addEventListener("click", () => {
    if (playTimer) { stopPlayback(); return; }
    setFrame(0);
    $("play").textContent = "Ⅱ Pause";
    playTimer = setInterval(() => {
      setFrame(frame + 1);
      if (frame === 4) stopPlayback();
    }, 850);
  });
  $("pixel-toggle").addEventListener("click", (event) => {
    const active = document.body.classList.toggle("pixelated");
    event.target.setAttribute("aria-pressed", String(active));
    event.target.textContent = active ? "Smooth display" : "Show native pixels";
  });
  loadTrajectory();

  // A typographic view of the same recorded pixels, with a literal-pixel toggle.
  const specimenCanvas = $("type-canvas");
  const specimenContext = specimenCanvas.getContext("2d");
  const sourceCanvas = document.createElement("canvas");
  sourceCanvas.width = sourceCanvas.height = 64;
  const sourceContext = sourceCanvas.getContext("2d", { willReadFrequently: true });
  let specimenIndex = 0;
  let specimenFrame = 4;
  let specimenType = true;
  let specimenTimer = null;
  let specimenImage = null;
  let specimenRequest = 0;
  const specimenSamples = data.gallery.samples.filter((sample) => sample.seed === 42);
  function drawSpecimen() {
    if (!specimenImage) return;
    sourceContext.clearRect(0, 0, 64, 64);
    sourceContext.drawImage(specimenImage, specimenFrame * 64, 0, 64, 64, 0, 0, 64, 64);
    specimenContext.clearRect(0, 0, 480, 480);
    if (!specimenType) {
      specimenContext.imageSmoothingEnabled = false;
      specimenContext.drawImage(sourceCanvas, 64, 64, 352, 352);
    } else {
      const pixels = sourceContext.getImageData(0, 0, 64, 64).data;
      const alphabet = " .·:+*ox#@";
      specimenContext.font = "8px Plex, monospace";
      specimenContext.textAlign = "center";
      for (let y = 0; y < 64; y++) {
        for (let x = 0; x < 64; x++) {
          const i = (y * 64 + x) * 4;
          const r = pixels[i], g = pixels[i + 1], b = pixels[i + 2];
          const brightness = (r + g + b) / 3;
          if (brightness > 244) continue;
          const glyph = alphabet[Math.min(9, Math.floor((255 - brightness) / 255 * 9) + 1)];
          specimenContext.fillStyle = "rgb(" + Math.round(r * .54) + "," + Math.round(g * .54) + "," + Math.round(b * .54) + ")";
          specimenContext.fillText(glyph, x * 7.5 + 3.75, y * 7.5 + 7);
        }
      }
    }
    const sample = specimenSamples[specimenIndex];
    $("specimen-caption").textContent = sample.species.toLowerCase() + " · seed 42 · " +
      (specimenFrame ? specimenFrame + " calls" : "initial noise") + (specimenType ? " · typeset from pixels" : " · actual pixels");
    specimenCanvas.setAttribute("aria-label", (specimenType ? "Typographic rendering" : "Recorded pixels") +
      " of " + sample.species + " after " + specimenFrame + " model calls");
  }
  function stopSpecimen() {
    clearInterval(specimenTimer);
    specimenTimer = null;
    $("specimen-play").textContent = "↺ replay";
  }
  function loadSpecimen() {
    stopSpecimen();
    const request = ++specimenRequest;
    const image = new Image();
    image.onload = () => {
      if (request !== specimenRequest) return;
      specimenImage = image;
      specimenFrame = 4;
      drawSpecimen();
    };
    image.src = "assets/gallery/" + specimenSamples[specimenIndex].trajectory;
  }
  $("specimen-play").addEventListener("click", () => {
    if (specimenTimer) { stopSpecimen(); return; }
    specimenFrame = 0; drawSpecimen();
    $("specimen-play").textContent = "Ⅱ pause";
    specimenTimer = setInterval(() => {
      specimenFrame++; drawSpecimen();
      if (specimenFrame === 4) stopSpecimen();
    }, 800);
  });
  $("specimen-mode").addEventListener("click", (event) => {
    specimenType = !specimenType;
    event.target.setAttribute("aria-pressed", String(specimenType));
    drawSpecimen();
  });
  $("specimen-next").addEventListener("click", () => {
    specimenIndex = (specimenIndex + 1) % specimenSamples.length;
    loadSpecimen();
  });
  specimenSamples.forEach((sample) => {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-label", "Explore " + sample.species);
    const image = new Image();
    image.src = "assets/gallery/" + sample.image;
    image.alt = "";
    image.width = image.height = 64;
    button.append(image, textElement("span", sample.species.toLowerCase()));
    button.addEventListener("click", () => {
      $("species").value = sample.species;
      loadTrajectory();
    });
    $("sample-shelf").append(button);
  });
  loadSpecimen();
  document.fonts.ready.then(drawSpecimen);

  let trailEnabled = !reducedMotion.matches;
  let trailIndex = 0;
  let trailPoint = null;
  function updateTrailControl() {
    $("trail-toggle").textContent = trailEnabled ? "··· trail on" : "··· trail off";
    $("trail-toggle").setAttribute("aria-pressed", String(!trailEnabled));
    $("trail-toggle").setAttribute("aria-label", trailEnabled ? "Turn sample trail off" : "Turn sample trail on");
    if (!trailEnabled) $("sample-trail").replaceChildren();
  }
  $("trail-toggle").addEventListener("click", () => { trailEnabled = !trailEnabled; updateTrailControl(); });
  $("top").addEventListener("pointermove", (event) => {
    if (!trailEnabled || reducedMotion.matches || event.pointerType !== "mouse" ||
        event.target.closest("a,button,canvas,.intro-note,.intro-title")) return;
    const bounds = $("top").getBoundingClientRect();
    const point = [event.clientX - bounds.left, event.clientY - bounds.top];
    if (trailPoint && Math.hypot(point[0] - trailPoint[0], point[1] - trailPoint[1]) < 95) return;
    trailPoint = point;
    const sample = specimenSamples[trailIndex % specimenSamples.length];
    const image = new Image();
    image.src = "assets/gallery/" + sample.image;
    image.className = "trail-sample";
    image.style.left = (point[0] - 24) + "px";
    image.style.top = (point[1] - 24) + "px";
    image.style.rotate = ((trailIndex % 5) * 7 - 14) + "deg";
    trailIndex++;
    $("sample-trail").append(image);
    if ($("sample-trail").children.length > 10) $("sample-trail").firstElementChild.remove();
    image.addEventListener("animationend", () => image.remove(), { once: true });
  });
  updateTrailControl();

  const diagram = document.querySelector(".flow-diagram > svg");
  let diagramPaused = reducedMotion.matches;
  function setDiagramMotion() {
    if (diagramPaused) diagram.pauseAnimations(); else diagram.unpauseAnimations();
    $("motion-toggle").textContent = diagramPaused ? "Play diagram motion" : "Pause diagram motion";
    $("motion-toggle").setAttribute("aria-pressed", String(diagramPaused));
  }
  $("motion-toggle").addEventListener("click", () => { diagramPaused = !diagramPaused; setDiagramMotion(); });
  reducedMotion.addEventListener("change", () => { diagramPaused = reducedMotion.matches; setDiagramMotion(); });
  setDiagramMotion();

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
  const pairChart = $("paired-chart");
  const px = (value) => 90 + (value - 19) / 1.1 * 345;
  [19, 19.25, 19.5, 19.75, 20].forEach((value) => {
    pairChart.append(svg("line", { x1: px(value), x2: px(value), y1: 40, y2: 228, stroke: "#e1ded3" }),
      svg("text", { x: px(value), y: 253, "text-anchor": "middle", class: "chart-axis" }, value.toFixed(2)));
  });
  data.pairs.forEach((row, index) => {
    const y = 55 + index * 39;
    pairChart.append(svg("text", { x: 64, y: y + 4, "text-anchor": "end", class: "chart-axis" }, `run ${index + 1}`),
      svg("line", { x1: px(row.tuned), x2: px(row.baseline), y1: y, y2: y, stroke: "#afb5a9", "stroke-width": 2 }),
      svg("circle", { cx: px(row.baseline), cy: y, r: 5, fill: "#8799a5" }),
      svg("circle", { cx: px(row.tuned), cy: y, r: 5, fill: "#47675b" }));
  });
  pairChart.append(svg("text", { x: 90, y: 282, fill: "#47675b", "font-size": 11 }, "● tuned"),
    svg("text", { x: 185, y: 282, fill: "#8799a5", "font-size": 11 }, "● ρ = 2"));

  if (data.study) {
    const panel = $("checkpoint-comparison");
    panel.replaceChildren();
    let revealed = false;
    const controls = textElement("div", "", "comparison-controls");
    const seedLabel = textElement("label", "Noise seed ");
    const seeds = document.createElement("select");
    seeds.setAttribute("aria-label", "Checkpoint comparison seed");
    data.study.visual_seeds.forEach((seed) => seeds.add(new Option(String(seed), String(seed))));
    seedLabel.append(seeds);
    const speciesLabel = textElement("label", "View ");
    const compareSpecies = document.createElement("select");
    compareSpecies.setAttribute("aria-label", "Checkpoint comparison species");
    compareSpecies.add(new Option("All 12 species", "all"));
    data.study.species.forEach((name) => compareSpecies.add(new Option(name, name.toLowerCase())));
    speciesLabel.append(compareSpecies);
    const reveal = textElement("button", "Reveal FID & steps", "button secondary");
    reveal.type = "button";
    reveal.setAttribute("aria-expanded", "false");
    controls.append(seedLabel, speciesLabel, reveal);
    const pairs = textElement("div", "", "compare-images");
    const comparisonFigures = data.study.checkpoints.map((checkpoint, index) => {
      const figure = document.createElement("figure");
      const image = document.createElement("img");
      image.loading = "lazy";
      const caption = document.createElement("figcaption");
      figure.append(image, caption); pairs.append(figure);
      return { checkpoint, image, caption, index };
    });
    function showComparison() {
      pairs.classList.toggle("single", compareSpecies.value !== "all");
      comparisonFigures.forEach(({ checkpoint, image, caption, index }) => {
        const speciesPart = compareSpecies.value === "all" ? "" : `${compareSpecies.value}-`;
        image.src = `assets/study/compare-${checkpoint.step}-${speciesPart}${seeds.value}.png`;
        image.alt = `Checkpoint ${index ? "B" : "A"}, ${compareSpecies.options[compareSpecies.selectedIndex].text}, seed ${seeds.value}`;
        image.style.aspectRatio = compareSpecies.value === "all" ? "4 / 3" : "1";
        const score = checkpoint.remeasured_uniform_fid ?? checkpoint.historical_uniform_fid;
        caption.replaceChildren(textElement("strong", revealed ? `${checkpoint.step / 1000}k updates` : `Checkpoint ${index ? "B" : "A"}`),
          textElement("span", revealed ? `${score.toFixed(4)} FID · uniform 4 calls` : "Look first. Reveal the score."));
      });
      reveal.textContent = revealed ? "Hide FID & steps" : "Reveal FID & steps";
      reveal.setAttribute("aria-expanded", String(revealed));
    }
    seeds.addEventListener("change", showComparison);
    compareSpecies.addEventListener("change", showComparison);
    reveal.addEventListener("click", () => { revealed = !revealed; showComparison(); });
    panel.append(controls, pairs);
    const findings = textElement("div", "", "compare-findings");
    findings.append(textElement("p", "64k is the first recorded checkpoint to cross FID 22. The 101k weights are the best recorded checkpoint in the same run. These comparisons use equal four-call intervals to isolate the checkpoint change."),
      textElement("p", "The headline 19.35 uses the tuned schedule and five seeds. It is a separate result from the uniform-schedule scores here. All 36 matched samples per checkpoint are shown; this is a qualitative study, not a human-preference benchmark."));
    panel.append(findings);
    showComparison();
    const journey = textElement("div", "", "training-journey");
    journey.append(textElement("h3", "One Pikachu, at six checkpoints."));
    const journeyGrid = textElement("div", "", "journey-grid");
    [1000, 8000, 32000, 64000, 101000, 122000].forEach((step) => {
      const figure = document.createElement("figure");
      const image = document.createElement("img");
      image.src = `assets/study/journey-${step}.png`;
      image.alt = `Pikachu at ${step.toLocaleString()} training updates, seed 42, uniform four-call sampler`;
      image.loading = "lazy"; image.width = image.height = 64;
      figure.append(image, textElement("figcaption", `${step / 1000}k updates`)); journeyGrid.append(figure);
    });
    journey.append(journeyGrid, textElement("p", "One fixed seed and class across six actual EMA checkpoints. These are training snapshots, not interpolated images.", "caption"));
    panel.append(journey);
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
    panel.append(lossSvg, textElement("p", "Mean logged adaptive loss in 1,000-update bins; shaded band shows the minimum and maximum logged loss in each bin, not a confidence interval. The objective has a changing target and adaptive weighting, so its scale is not an image-quality score. Duplicate logged steps retain their last occurrence.", "caption"));
    const link = textElement("a", "Download the numeric loss history ↗", "caption");
    link.href = "assets/study/training_loss.csv"; panel.append(link);
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
  window.addEventListener("resize", () => { drawFID(); drawLoss(); });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { stopPlayback(); stopChart(); stopSpecimen(); diagram.pauseAnimations(); }
    else setDiagramMotion();
  });
})();
