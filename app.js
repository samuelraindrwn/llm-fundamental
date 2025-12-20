// ==========================================
// BELAJAR AI DARI NOL - STORYTELLING APP
// ==========================================

// ==========================================
// 1. DOM ELEMENTS
// ==========================================
const elements = {
  // Hero
  startBtn: document.getElementById("startBtn"),
  scrollHint: document.getElementById("scrollHint"),
  storyContainer: document.getElementById("storyContainer"),

  // Inputs
  trainingText: document.getElementById("trainingText"),
  epochs: document.getElementById("epochs"),
  learningRate: document.getElementById("learningRate"),
  startWord: document.getElementById("startWord"),

  // Step 1
  vocabGrid: document.getElementById("vocabGrid"),
  tokenizationViz: document.getElementById("tokenizationViz"),
  vocabNoteText: document.getElementById("vocabNoteText"),

  // Step 2
  embeddingTable: document.getElementById("embeddingTable"),

  // Step 3
  trainingData: document.getElementById("trainingData"),
  dataCount: document.getElementById("dataCount"),

  // Step 4
  inputLayer: document.getElementById("inputLayer"),
  hiddenLayer: document.getElementById("hiddenLayer"),
  outputLayer: document.getElementById("outputLayer"),
  outputSize: document.getElementById("outputSize"),
  totalParams: document.getElementById("totalParams"),
  totalParamsInsight: document.getElementById("totalParamsInsight"),
  vocabSizeDisplay: document.getElementById("vocabSizeDisplay"),
  vocabSizeInsight: document.getElementById("vocabSizeInsight"),

  // Step 5
  forwardPassInput: document.getElementById("forwardPassInput"),
  fpInputChars: document.getElementById("fpInputChars"),
  fpEmbedding: document.getElementById("fpEmbedding"),
  fpHidden: document.getElementById("fpHidden"),
  fpOutput: document.getElementById("fpOutput"),
  fpPrediction: document.getElementById("fpPrediction"),

  // Step 6
  trainBtn: document.getElementById("trainBtn"),
  stopBtn: document.getElementById("stopBtn"),
  currentEpoch: document.getElementById("currentEpoch"),
  currentLoss: document.getElementById("currentLoss"),
  trainingStatus: document.getElementById("trainingStatus"),
  progressFill: document.getElementById("progressFill"),
  lossChart: document.getElementById("lossChart"),
  trainingComplete: document.getElementById("trainingComplete"),

  // Step 7
  generateStatus: document.getElementById("generateStatus"),
  generateBtn: document.getElementById("generateBtn"),
  generatedText: document.getElementById("generatedText"),
  genStepsList: document.getElementById("genStepsList"),
};

// ==========================================
// 2. GLOBAL STATE
// ==========================================
let state = {
  isTraining: false,
  isTrained: false,
  stopRequested: false,

  chars: [],
  vocabSize: 0,
  stoi: {},
  itos: {},

  data: [],
  xData: [],
  yData: [],
  xTrain: [],

  nEmbd: 4,
  nHidden: 48,
  blockSize: 3,
  epochs: 2000,
  lr: 0.05,
  inputDim: 0,

  embeddingTable: [],
  positionEmbeddingTable: [],
  W1: [],
  W2: [],
  b1: [],
  b2: [],

  lossHistory: [],
};

let chart = null;

// ==========================================
// 3. UTILITY FUNCTIONS
// ==========================================
let seed = 42;

function seededRandom() {
  seed = (seed * 9301 + 49297) % 233280;
  return seed / 233280;
}

function gauss(mean = 0, std = 1) {
  let u = 0,
    v = 0;
  while (u === 0) u = seededRandom();
  while (v === 0) v = seededRandom();
  return (
    mean + std * Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
  );
}

function dot(v1, v2) {
  let sum = 0;
  for (let i = 0; i < v1.length; i++) {
    sum += v1[i] * v2[i];
  }
  return sum;
}

function outer(v1, v2) {
  return v1.map((x) => v2.map((y) => x * y));
}

function relu(v) {
  return v.map((x) => Math.max(0, x));
}

function softmax(logits) {
  const m = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / s);
}

function transpose(matrix) {
  return matrix[0].map((_, i) => matrix.map((row) => row[i]));
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getColorForValue(val) {
  const intensity = Math.min(Math.abs(val) * 3, 1);
  if (val >= 0) {
    return `rgba(99, 102, 241, ${0.3 + intensity * 0.7})`;
  } else {
    return `rgba(236, 72, 153, ${0.3 + intensity * 0.7})`;
  }
}

function formatNumber(n, decimals = 2) {
  return n.toFixed(decimals);
}

// ==========================================
// 4. MODEL INITIALIZATION
// ==========================================
function getTrainingText() {
  // Check if training text input exists
  const trainingTextEl = document.getElementById("trainingText");
  let text = trainingTextEl ? trainingTextEl.value : "aku suka koding ai.";

  // Force lowercase to avoid case sensitivity issues
  text = text.toLowerCase();

  // Auto-add period if missing (prevents infinite generation loop)
  if (text.length > 0 && !text.endsWith(".")) {
    text = text + ".";
  }

  return text;
}

function initializeModel() {
  seed = 42;

  const text = getTrainingText();

  state.chars = [...new Set(text)].sort();
  state.vocabSize = state.chars.length;
  state.stoi = {};
  state.itos = {};

  state.chars.forEach((ch, i) => {
    state.stoi[ch] = i;
    state.itos[i] = ch;
  });

  state.data = text.split("").map((c) => state.stoi[c]);
  state.xData = [];
  state.yData = [];

  for (let i = 0; i < state.data.length - state.blockSize; i++) {
    state.xData.push(state.data.slice(i, i + state.blockSize));
    state.yData.push(state.data[i + state.blockSize]);
  }

  state.inputDim = state.nEmbd * state.blockSize;

  const scaleW1 = Math.sqrt(2 / state.inputDim);
  const scaleW2 = Math.sqrt(2 / state.nHidden);

  state.embeddingTable = Array.from({ length: state.vocabSize }, () =>
    Array.from({ length: state.nEmbd }, () => gauss(0, 0.1))
  );

  state.positionEmbeddingTable = Array.from({ length: state.blockSize }, () =>
    Array.from({ length: state.nEmbd }, () => gauss(0, 0.1))
  );

  state.W1 = Array.from({ length: state.inputDim }, () =>
    Array.from({ length: state.nHidden }, () => gauss(0, scaleW1))
  );

  state.W2 = Array.from({ length: state.nHidden }, () =>
    Array.from({ length: state.vocabSize }, () => gauss(0, scaleW2))
  );

  state.b1 = Array(state.nHidden).fill(0);
  state.b2 = Array(state.vocabSize).fill(0);

  state.xTrain = state.xData.map((sequence) => {
    const seqVecs = [];
    sequence.forEach((idx, t) => {
      for (let j = 0; j < state.nEmbd; j++) {
        seqVecs.push(
          state.embeddingTable[idx][j] + state.positionEmbeddingTable[t][j]
        );
      }
    });
    return seqVecs;
  });

  state.lossHistory = [];
}

// ==========================================
// 5. RENDER FUNCTIONS
// ==========================================

function renderVocab() {
  const text = elements.trainingText.value;
  const chars = [...new Set(text)].sort();

  let html = "";
  chars.forEach((char, idx) => {
    const displayChar = char === " " ? "␣" : char;
    html += `
            <div class="vocab-item" style="animation-delay: ${idx * 0.05}s">
                <span class="char">${displayChar}</span>
                <span class="idx">ID: ${idx}</span>
            </div>
        `;
  });
  elements.vocabGrid.innerHTML = html;

  // Dynamic Vocab Note
  if (elements.vocabNoteText && chars.length > 0) {
    if (chars.includes(" ")) {
      elements.vocabNoteText.innerHTML =
        "Spasi berkontribusi sebagai karakter dengan ID = " +
        chars.indexOf(" ") +
        ". Kenapa " +
        chars.indexOf(" ") +
        "? Karena dia paling awal urutan abjadnya.";
    } else {
      const firstChar = chars[0];
      const displayFirst = firstChar === " " ? "Spasi" : `"${firstChar}"`;
      elements.vocabNoteText.innerHTML = `Karakter ${displayFirst} dapet ID = 0 karena dia paling awal di urutan abjad/ASCII.`;
    }
  }
}

function updateCodeSnippets() {
  const text = getTrainingText();
  const chars = [...new Set(text)].sort();

  // Update code text references
  const codeText1 = document.getElementById("codeText1");
  if (codeText1) codeText1.textContent = text;

  // Update chars array display
  const codeChars1 = document.getElementById("codeChars1");
  if (codeChars1) {
    const charsDisplay = chars
      .map((c) => (c === " " ? "' '" : `'${c}'`))
      .join(", ");
    codeChars1.textContent = `[${charsDisplay}]`;
  }

  // Update stoi display
  const codeStoi1 = document.getElementById("codeStoi1");
  if (codeStoi1) {
    const stoiDisplay = chars
      .slice(0, 4)
      .map((c, i) => {
        const char = c === " " ? "' '" : `'${c}'`;
        return `${char}: ${i}`;
      })
      .join(", ");
    codeStoi1.textContent = `{${stoiDisplay}, ...}`;
  }
}

function renderTokenization() {
  const text = elements.trainingText.value;
  initializeModel();

  let html = "";
  text.split("").forEach((char, i) => {
    const displayChar = char === " " ? "␣" : char;
    const id = state.stoi[char];
    html += `
            <div class="token-item" style="animation-delay: ${i * 0.05}s">
                <div class="token-char">${displayChar}</div>
                <div class="token-arrow">↓</div>
                <div class="token-id">${id}</div>
            </div>
        `;
  });
  elements.tokenizationViz.innerHTML = html;
}

function renderEmbedding() {
  let html = "";
  state.chars.forEach((char, idx) => {
    const displayChar = char === " " ? "␣" : char;
    const values = state.embeddingTable[idx];
    html += `
            <div class="embedding-row">
                <div class="embedding-label">${displayChar}</div>
                <div class="embedding-values">
                    ${values
                      .map(
                        (v) => `
                        <div class="embedding-cell" style="background: ${getColorForValue(
                          v
                        )}" title="${formatNumber(v, 3)}">
                            ${formatNumber(v)}
                        </div>
                    `
                      )
                      .join("")}
                </div>
            </div>
        `;
  });
  elements.embeddingTable.innerHTML = html;
}

function renderTrainingData() {
  let html = `
    <table class="training-table">
      <thead>
        <tr>
          <th>#</th>
          <th>Input (3 huruf)</th>
          <th></th>
          <th>Target</th>
        </tr>
      </thead>
      <tbody>
  `;

  state.xData.forEach((input, i) => {
    const inputChars = input.map((idx) => state.itos[idx]).join("");
    const targetChar = state.itos[state.yData[i]];
    const displayInput = inputChars.replace(/ /g, "␣");
    const displayTarget = targetChar === " " ? "␣" : targetChar;

    html += `
      <tr>
        <td class="row-num">${i + 1}</td>
        <td class="input-cell">"${displayInput}"</td>
        <td class="arrow-cell">→</td>
        <td class="target-cell">"${displayTarget}"</td>
      </tr>
    `;
  });

  html += `</tbody></table>`;
  elements.trainingData.innerHTML = html;
  elements.dataCount.textContent = state.xData.length;
}

function renderNetwork() {
  // Input neurons
  let inputHtml = "";
  for (let i = 0; i < state.inputDim; i++) {
    inputHtml += '<div class="neuron"></div>';
  }
  elements.inputLayer.innerHTML = inputHtml;

  // Hidden neurons
  let hiddenHtml = "";
  for (let i = 0; i < state.nHidden; i++) {
    hiddenHtml += '<div class="neuron"></div>';
  }
  elements.hiddenLayer.innerHTML = hiddenHtml;

  // Output neurons
  let outputHtml = "";
  for (let i = 0; i < state.vocabSize; i++) {
    outputHtml += '<div class="neuron"></div>';
  }
  elements.outputLayer.innerHTML = outputHtml;

  // Update stats
  elements.outputSize.textContent = `${state.vocabSize} neuron`;
  elements.vocabSizeDisplay.textContent = state.vocabSize;
  if (elements.vocabSizeInsight)
    elements.vocabSizeInsight.textContent = state.vocabSize;

  const w1Params = state.inputDim * state.nHidden;
  const w2Params = state.nHidden * state.vocabSize;
  const biasParams = state.nHidden + state.vocabSize;
  const embParams =
    state.vocabSize * state.nEmbd + state.blockSize * state.nEmbd;
  const total = w1Params + w2Params + biasParams + embParams;
  const formattedTotal = total.toLocaleString();

  if (elements.totalParams) elements.totalParams.textContent = formattedTotal;
  if (elements.totalParamsInsight)
    elements.totalParamsInsight.textContent = formattedTotal;
}

function renderForwardPassSetup() {
  let optionsHtml = "";
  state.xData.forEach((input, i) => {
    const chars = input.map((idx) => state.itos[idx]).join("");
    optionsHtml += `<option value="${i}">"${chars}"</option>`;
  });
  elements.forwardPassInput.innerHTML = optionsHtml;
  renderForwardPass(0, false);
}

async function renderForwardPass(dataIdx, animate = true) {
  const input = state.xData[dataIdx];
  const inputChars = input.map((idx) => state.itos[idx]);

  // Step 1: Input chars
  let inputHtml = "";
  inputChars.forEach((c, i) => {
    inputHtml += `<div class="flow-char" style="animation-delay: ${i * 0.1}s">${
      c === " " ? "␣" : c
    }</div>`;
  });
  elements.fpInputChars.innerHTML = inputHtml;

  if (animate) {
    document.getElementById("flowStep1").classList.add("active");
    await sleep(400);
  }

  // Step 2: Embedding
  const embVec = [];
  input.forEach((idx, t) => {
    for (let j = 0; j < state.nEmbd; j++) {
      embVec.push(
        state.embeddingTable[idx][j] + state.positionEmbeddingTable[t][j]
      );
    }
  });

  let embHtml = "";
  embVec.forEach((v, i) => {
    embHtml += `<div class="flow-val" style="animation-delay: ${
      i * 0.02
    }s">${formatNumber(v)}</div>`;
  });
  elements.fpEmbedding.innerHTML = embHtml;

  if (animate) {
    document.getElementById("flowStep1").classList.remove("active");
    document.getElementById("flowStep2").classList.add("active");
    await sleep(100);
    document
      .querySelectorAll("#fpEmbedding .flow-val")
      .forEach((el) => el.classList.add("active"));
    await sleep(500);
  }

  // Step 3: Hidden layer
  const colsW1 = transpose(state.W1);
  const hPre = colsW1.map((col, j) => dot(embVec, col) + state.b1[j]);
  const h = relu(hPre);

  let hiddenHtml = "";
  h.forEach((v, i) => {
    const isActive = v > 0;
    hiddenHtml += `<div class="flow-neuron ${
      isActive ? "active" : ""
    }" title="${formatNumber(v)}"></div>`;
  });
  elements.fpHidden.innerHTML = hiddenHtml;

  if (animate) {
    document.getElementById("flowStep2").classList.remove("active");
    document.getElementById("flowStep3").classList.add("active");
    await sleep(600);
  }

  // Step 4: Output probabilities
  const colsW2 = transpose(state.W2);
  const logits = colsW2.map((col, j) => dot(h, col) + state.b2[j]);
  const probs = softmax(logits);

  const sortedProbs = state.chars
    .map((char, idx) => ({
      char: char === " " ? "␣" : char,
      prob: probs[idx],
      idx: idx,
    }))
    .sort((a, b) => b.prob - a.prob);

  let probHtml = "";
  sortedProbs.forEach((item, i) => {
    probHtml += `
            <div class="flow-prob-item">
                <span class="flow-prob-char">${item.char}</span>
                <div class="flow-prob-bar">
                    <div class="flow-prob-fill ${
                      i === 0 ? "top" : ""
                    }" style="width: ${item.prob * 100}%"></div>
                </div>
                <span class="flow-prob-val">${(item.prob * 100).toFixed(
                  1
                )}%</span>
            </div>
        `;
  });
  elements.fpOutput.innerHTML = probHtml;

  if (animate) {
    document.getElementById("flowStep3").classList.remove("active");
    document.getElementById("flowStep4").classList.add("active");
    await sleep(400);
  }

  // Step 5: Prediction
  const bestChar = sortedProbs[0].char;
  elements.fpPrediction.innerHTML = `<span class="pred-char">${bestChar}</span>`;

  if (animate) {
    document.getElementById("flowStep4").classList.remove("active");
    document.getElementById("flowStep5").classList.add("active");
    await sleep(300);
    document.getElementById("flowStep5").classList.remove("active");
  }
}

// ==========================================
// 6. CHART
// ==========================================
function initChart() {
  if (chart) {
    chart.destroy();
  }

  const ctx = elements.lossChart.getContext("2d");

  chart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Loss",
          data: [],
          borderColor: "#6366f1",
          backgroundColor: "rgba(99, 102, 241, 0.1)",
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 0 },
      scales: {
        x: {
          display: true,
          title: { display: true, text: "Epoch", color: "#a1a1aa" },
          ticks: { color: "#71717a" },
          grid: { color: "rgba(255,255,255,0.05)" },
        },
        y: {
          display: true,
          title: { display: true, text: "Loss", color: "#a1a1aa" },
          ticks: { color: "#71717a" },
          grid: { color: "rgba(255,255,255,0.05)" },
          beginAtZero: true,
        },
      },
      plugins: { legend: { display: false } },
    },
  });
}

// ==========================================
// 7. TRAINING
// ==========================================
async function train() {
  state.isTraining = true;
  state.stopRequested = false;
  state.isTrained = false;

  state.epochs = parseInt(elements.epochs.value);
  state.lr = parseFloat(elements.learningRate.value);

  initializeModel();
  initChart();

  elements.trainBtn.disabled = true;
  elements.stopBtn.disabled = false;
  elements.trainBtn.classList.add("training-active");
  elements.trainingStatus.textContent = "Training...";
  elements.trainingComplete.style.display = "none";

  const updateInterval = Math.max(1, Math.floor(state.epochs / 100));

  for (let step = 0; step < state.epochs; step++) {
    if (state.stopRequested) break;

    let lossSum = 0;

    for (let i = 0; i < state.xTrain.length; i++) {
      const inputVec = state.xTrain[i];
      const targetIdx = state.yData[i];

      const colsW1 = transpose(state.W1);
      const hPre = colsW1.map((col, j) => dot(inputVec, col) + state.b1[j]);
      const h = relu(hPre);

      const colsW2 = transpose(state.W2);
      const logits = colsW2.map((col, j) => dot(h, col) + state.b2[j]);

      const probs = softmax(logits);
      const loss = -Math.log(probs[targetIdx] + 1e-9);
      lossSum += loss;

      const dlogits = [...probs];
      dlogits[targetIdx] -= 1;

      const dW2 = outer(h, dlogits);
      const db2 = dlogits;

      const dh = state.W2.map((row, j) => dot(dlogits, row));
      const dhPre = dh.map((val, j) => (hPre[j] > 0 ? val : 0));

      const dW1 = outer(inputVec, dhPre);
      const db1 = dhPre;

      for (let r = 0; r < state.nHidden; r++) {
        for (let c = 0; c < state.vocabSize; c++) {
          state.W2[r][c] -= state.lr * dW2[r][c];
        }
      }

      for (let c = 0; c < state.vocabSize; c++) {
        state.b2[c] -= state.lr * db2[c];
      }

      for (let r = 0; r < state.inputDim; r++) {
        for (let c = 0; c < state.nHidden; c++) {
          state.W1[r][c] -= state.lr * dW1[r][c];
        }
      }

      for (let c = 0; c < state.nHidden; c++) {
        state.b1[c] -= state.lr * db1[c];
      }
    }

    const avgLoss = lossSum / state.xTrain.length;

    if (step % updateInterval === 0 || step === state.epochs - 1) {
      state.lossHistory.push({ epoch: step, loss: avgLoss });

      elements.currentEpoch.textContent = step;
      elements.currentLoss.textContent = avgLoss.toFixed(4);
      elements.progressFill.style.width = `${(step / state.epochs) * 100}%`;

      if (chart) {
        chart.data.labels.push(step);
        chart.data.datasets[0].data.push(avgLoss);
        chart.update("none");
      }

      await sleep(1);
    }
  }

  state.isTraining = false;
  state.isTrained = true;

  elements.trainBtn.disabled = false;
  elements.stopBtn.disabled = true;
  elements.trainBtn.classList.remove("training-active");
  elements.progressFill.style.width = "100%";
  elements.trainingStatus.textContent = "Selesai! ✅";
  elements.currentEpoch.textContent = state.epochs;
  elements.trainingComplete.style.display = "flex";

  // Enable generate
  updateGenerateStatus();

  // Enable exploration mode
  enableExplorationMode();
}

function stopTraining() {
  state.stopRequested = true;
}

// ==========================================
// 8. GENERATE
// ==========================================
function updateGenerateStatus() {
  if (state.isTrained) {
    elements.generateStatus.innerHTML = `
      <div class="success-banner">
        <span class="success-icon">🎉</span>
        <div class="success-text">
          <strong>AI sudah dilatih!</strong>
          <span>Sekarang kita bisa generate teks.</span>
        </div>
      </div>`;
    elements.generateStatus.classList.add("trained");
    elements.generateBtn.disabled = false;
  } else {
    elements.generateStatus.innerHTML =
      "<p>⚠️ <strong>Tunggu!</strong> Training dulu di Bab 6 sebelum generate.</p>";
    elements.generateStatus.classList.remove("trained");
    elements.generateBtn.disabled = true;
  }
}

async function generate() {
  if (!state.isTrained) return;

  // Force lowercase to match training vocabulary
  const startWord = elements.startWord.value.toLowerCase();

  for (const char of startWord) {
    if (!(char in state.stoi)) {
      elements.generatedText.innerHTML = `<span class="placeholder">Error: Karakter "${char}" tidak ada di vocabulary!</span>`;
      return;
    }
  }

  let currIdx = startWord.split("").map((c) => state.stoi[c]);
  let result = startWord;

  elements.generatedText.innerHTML = `<span class="char">${startWord}</span>`;
  elements.genStepsList.innerHTML = "";

  const genSteps = [];

  for (let i = 0; i < 50; i++) {
    const inputSeq = currIdx.slice(-state.blockSize);
    const inputChars = inputSeq.map((idx) => state.itos[idx]).join("");

    const flatVec = [];
    inputSeq.forEach((idx, t) => {
      for (let j = 0; j < state.nEmbd; j++) {
        flatVec.push(
          state.embeddingTable[idx][j] + state.positionEmbeddingTable[t][j]
        );
      }
    });

    const colsW1 = transpose(state.W1);
    const hPre = colsW1.map((col, j) => dot(flatVec, col) + state.b1[j]);
    const h = relu(hPre);

    const colsW2 = transpose(state.W2);
    const logits = colsW2.map((col, j) => dot(h, col) + state.b2[j]);

    const probs = softmax(logits);
    const bestIdx = probs.indexOf(Math.max(...probs));
    const bestProb = probs[bestIdx];

    const char = state.itos[bestIdx];
    const displayChar = char === " " ? "␣" : char;

    genSteps.push({
      input: inputChars,
      output: displayChar,
      prob: bestProb,
    });

    // Stop if we hit period
    if (char === ".") break;

    result += char;
    currIdx.push(bestIdx);

    // Loop detection: check if result ends with repeating pattern
    if (result.length > 10) {
      const lastChunk = result.slice(-8);
      const beforeChunk = result.slice(-16, -8);
      if (lastChunk === beforeChunk) {
        // Detected loop! Stop and add warning
        elements.generatedText.innerHTML += `<span class="char warning"> [Loop detected - stopped]</span>`;
        break;
      }
    }

    elements.generatedText.innerHTML += `<span class="char new">${
      char === " " ? "&nbsp;" : char
    }</span>`;

    await sleep(150);
  }

  let stepsHtml = "";
  genSteps.forEach((step, i) => {
    stepsHtml += `
            <div class="gen-step-item" style="animation-delay: ${i * 0.05}s">
                <span class="gen-step-input">"${step.input}"</span>
                <span class="gen-step-arrow">→</span>
                <span class="gen-step-output">${step.output}</span>
                <span class="gen-step-prob">(${(step.prob * 100).toFixed(
                  1
                )}%)</span>
            </div>
        `;
  });
  elements.genStepsList.innerHTML = stepsHtml;
}

// ==========================================
// 9. INTERSECTION OBSERVER FOR ANIMATIONS
// ==========================================

// Reveal animations for elements as they scroll into view
function setupRevealAnimations() {
  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
        }
      });
    },
    { threshold: 0.15, rootMargin: "0px 0px -50px 0px" }
  );

  // Observe all reveal elements
  document
    .querySelectorAll(
      ".reveal, .reveal-left, .reveal-right, .reveal-scale, .stagger-children"
    )
    .forEach((el) => {
      revealObserver.observe(el);
    });
}

// Chapter-specific content loading
function setupChapterObserver() {
  const chapterObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const chapter = entry.target;
          const chapterId = chapter.id;

          // Add visible class for animations
          chapter.classList.add("chapter-visible");

          // Trigger render based on chapter
          switch (chapterId) {
            case "chapter1":
              renderVocab();
              renderTokenization();
              break;
            case "chapter2":
              renderEmbedding();
              break;
            case "chapter3":
              renderTrainingData();
              break;
            case "chapter4":
              renderNetwork();
              break;
            case "chapter5":
              renderForwardPassSetup();
              break;
            case "chapter6":
              initChart();
              break;
            case "chapter7":
              updateGenerateStatus();
              break;
          }
        }
      });
    },
    { threshold: 0.2 }
  );

  document.querySelectorAll(".chapter").forEach((chapter) => {
    chapterObserver.observe(chapter);
  });
}

// Parallax effect for hero section
function setupParallaxEffect() {
  const hero = document.querySelector(".hero");
  const heroContent = document.querySelector(".hero-content");

  window.addEventListener("scroll", () => {
    const scrollY = window.scrollY;
    const heroHeight = hero.offsetHeight;

    if (scrollY < heroHeight) {
      const opacity = 1 - (scrollY / heroHeight) * 1.5;
      const translateY = scrollY * 0.4;
      const scale = 1 - (scrollY / heroHeight) * 0.2;

      heroContent.style.opacity = Math.max(opacity, 0);
      heroContent.style.transform = `translateY(${translateY}px) scale(${Math.max(
        scale,
        0.8
      )})`;
    }
  });
}

// Chapter parallax disabled - was causing glitches
// Hero parallax is still active

// Scroll-triggered forward pass animation
function setupScrollTriggeredForwardPass() {
  let forwardPassAnimated = false;

  const forwardPassObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting && !forwardPassAnimated) {
          forwardPassAnimated = true;
          // Delay to let user see the section first
          setTimeout(() => {
            renderForwardPass(0, true);
          }, 500);
        }
      });
    },
    { threshold: 0.5 }
  );

  const forwardFlow = document.querySelector(".forward-flow");
  if (forwardFlow) {
    forwardPassObserver.observe(forwardFlow);
  }

  // Also trigger on dropdown change
  const dropdown = document.getElementById("forwardPassInput");
  if (dropdown) {
    dropdown.addEventListener("change", (e) => {
      renderForwardPass(parseInt(e.target.value), true);
    });
  }
}

// Progress indicator
function setupProgressIndicator() {
  const progressBar = document.createElement("div");
  progressBar.className = "reading-progress";
  progressBar.innerHTML = '<div class="reading-progress-fill"></div>';
  document.body.appendChild(progressBar);

  window.addEventListener("scroll", () => {
    const docHeight =
      document.documentElement.scrollHeight - window.innerHeight;
    const scrollPercent = (window.scrollY / docHeight) * 100;
    document.querySelector(
      ".reading-progress-fill"
    ).style.width = `${scrollPercent}%`;
  });
}

// Add reveal classes to elements
function addRevealClasses() {
  // Chapter headers get reveal animation
  document.querySelectorAll(".chapter-header").forEach((el) => {
    el.classList.add("reveal");
  });

  // Story text paragraphs
  document.querySelectorAll(".story-text").forEach((el) => {
    el.classList.add("reveal");
  });

  // Code blocks slide in from left
  document.querySelectorAll(".code-block").forEach((el) => {
    el.classList.add("reveal-left");
  });

  // Visual cards scale in
  document
    .querySelectorAll(".visual-card, .tokenization-card, .mission-card")
    .forEach((el) => {
      el.classList.add("reveal-scale");
    });

  // Insight boxes slide in from right
  document.querySelectorAll(".insight-box").forEach((el) => {
    el.classList.add("reveal-right");
  });

  // Network visual
  document.querySelectorAll(".network-visual").forEach((el) => {
    el.classList.add("reveal-scale");
  });

  // Training and generate panels
  document
    .querySelectorAll(".training-panel, .generate-panel, .demo-box")
    .forEach((el) => {
      el.classList.add("reveal");
    });

  // Comparison boxes
  document.querySelectorAll(".comparison-box").forEach((el) => {
    el.classList.add("reveal");
  });

  // Stats grid
  document.querySelectorAll(".stats-grid").forEach((el) => {
    el.classList.add("stagger-children");
  });

  // Recap grid
  document.querySelectorAll(".recap-grid").forEach((el) => {
    el.classList.add("stagger-children");
  });

  // Learning steps
  document.querySelectorAll(".learning-steps").forEach((el) => {
    el.classList.add("reveal");
  });

  // Input sections
  document.querySelectorAll(".input-section").forEach((el) => {
    el.classList.add("reveal");
  });
}

// ==========================================
// 10. EVENT LISTENERS
// ==========================================
function setupEventListeners() {
  // Hero button - smooth scroll to first chapter
  elements.startBtn.addEventListener("click", () => {
    document.getElementById("chapter0").scrollIntoView({ behavior: "smooth" });
  });

  // Training text input - real-time updates
  if (elements.trainingText) {
    elements.trainingText.addEventListener("input", () => {
      // Force lowercase in real-time
      const cursorPos = elements.trainingText.selectionStart;
      elements.trainingText.value = elements.trainingText.value.toLowerCase();
      elements.trainingText.setSelectionRange(cursorPos, cursorPos);

      const text = elements.trainingText.value;

      // Update stats
      const charCount = document.getElementById("charCount");
      const uniqueCount = document.getElementById("uniqueCount");
      if (charCount) charCount.textContent = text.length;
      if (uniqueCount) uniqueCount.textContent = new Set(text).size;

      // Reinitialize model and re-render
      initializeModel();
      state.isTrained = false;
      renderVocab();
      updateCodeSnippets();
      renderTokenization();
      renderEmbedding();
      renderTrainingData();
      renderNetwork();
      renderForwardPassSetup();
      updateGenerateStatus();
    });
  }

  // Training
  elements.trainBtn.addEventListener("click", train);
  elements.stopBtn.addEventListener("click", stopTraining);

  // Generate
  elements.generateBtn.addEventListener("click", generate);

  // Restart button - scroll to top and focus on input
  const restartBtn = document.getElementById("restartBtn");
  if (restartBtn) {
    restartBtn.addEventListener("click", () => {
      // Scroll to chapter 0 (input section)
      document
        .getElementById("chapter0")
        .scrollIntoView({ behavior: "smooth" });

      // Focus on input and select text
      setTimeout(() => {
        if (elements.trainingText) {
          elements.trainingText.focus();
          elements.trainingText.select();
        }
      }, 500);
    });
  }

  // Retrain button (in exploration mode)
  const retrainBtn = document.getElementById("retrainBtn");
  if (retrainBtn) {
    retrainBtn.addEventListener("click", () => {
      const trainingText = document.getElementById("trainingText");
      if (trainingText) {
        // Update display text
        const displayText = document.getElementById("displayText");
        if (displayText) {
          displayText.textContent = `"${trainingText.value}"`;
        }
        // Re-initialize and re-render everything
        initializeModel();
        state.isTrained = false;
        renderVocab();
        renderTokenization();
        renderEmbedding();
        renderTrainingData();
        renderNetwork();
        renderForwardPassSetup();
        updateGenerateStatus();
        // Scroll to training section
        document
          .getElementById("chapter6")
          .scrollIntoView({ behavior: "smooth" });
      }
    });
  }

  // Training text change handler (in exploration mode)
  if (elements.trainingText) {
    elements.trainingText.addEventListener("change", () => {
      // Only update if we're in exploration mode
      const explorationSection = document.getElementById("explorationSection");
      if (explorationSection && explorationSection.style.display !== "none") {
        initializeModel();
        state.isTrained = false;
      }
    });
  }
}

// Show exploration mode after training
function enableExplorationMode() {
  const explorationSection = document.getElementById("explorationSection");
  if (explorationSection) {
    explorationSection.style.display = "block";
  }
}

// ==========================================
// 11. INITIALIZATION
// ==========================================
function init() {
  // Use default training text
  const defaultText = "aku suka koding ai.";

  // Store in global for use
  window.trainingTextValue = defaultText;

  initializeModel();

  // Add animation classes first
  addRevealClasses();

  // Setup animations
  setupRevealAnimations();
  setupChapterObserver();
  setupParallaxEffect();
  setupProgressIndicator();
  setupScrollTriggeredForwardPass();

  // Setup event listeners
  setupEventListeners();

  // Initial renders for visible sections
  renderVocab();
  updateCodeSnippets();
  renderTokenization();
}

// Override training text getter to use default or exploration input
const originalTrainingTextValue = () => {
  const trainingTextEl = document.getElementById("trainingText");
  return trainingTextEl ? trainingTextEl.value : "aku suka koding ai.";
};

document.addEventListener("DOMContentLoaded", init);
