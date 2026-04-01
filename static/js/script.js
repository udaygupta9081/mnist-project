const canvas = document.getElementById("digit-canvas");
const context = canvas.getContext("2d");
const predictButton = document.getElementById("predict-button");
const clearButton = document.getElementById("clear-button");
const loader = document.getElementById("loader");
const resultCard = document.getElementById("result-card");
const errorCard = document.getElementById("error-card");
const digitResult = document.getElementById("digit-result");
const confidenceResult = document.getElementById("confidence-result");
const topPredictions = document.getElementById("top-predictions");
const processedPreview = document.getElementById("processed-preview");

const state = {
  drawing: false,
  hasInk: false,
  pointerId: null,
};

function fillCanvasBackground() {
  context.fillStyle = "#000000";
  context.fillRect(0, 0, canvas.width, canvas.height);
}

function configureCanvas() {
  fillCanvasBackground();
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = "#ffffff";
  context.lineWidth = 20;
}

function getPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * canvas.height,
  };
}

function startDrawing(event) {
  event.preventDefault();
  state.drawing = true;
  state.pointerId = event.pointerId;

  const point = getPoint(event);
  context.beginPath();
  context.moveTo(point.x, point.y);

  if (canvas.setPointerCapture) {
    canvas.setPointerCapture(event.pointerId);
  }
}

function draw(event) {
  if (!state.drawing || event.pointerId !== state.pointerId) {
    return;
  }

  event.preventDefault();
  const point = getPoint(event);
  context.lineTo(point.x, point.y);
  context.stroke();
  state.hasInk = true;
}

function stopDrawing(event) {
  if (event && state.pointerId !== null && event.pointerId !== state.pointerId) {
    return;
  }

  state.drawing = false;
  state.pointerId = null;
  context.beginPath();
}

function setLoading(isLoading) {
  loader.classList.toggle("hidden", !isLoading);
  loader.classList.toggle("flex", isLoading);
  predictButton.disabled = isLoading || !window.APP_CONFIG.modelReady;
  predictButton.textContent = isLoading ? "Predicting..." : "Predict Digit";
}

function showError(message) {
  errorCard.textContent = message;
  errorCard.classList.remove("hidden");
  resultCard.classList.add("hidden");
}

function hideError() {
  errorCard.classList.add("hidden");
  errorCard.textContent = "";
}

function renderPredictions(items) {
  topPredictions.innerHTML = "";

  if (!items.length) {
    const li = document.createElement("li");
    li.className = "prediction-chip";
    li.innerHTML = "<strong>Only label</strong><span>No probability scores</span>";
    topPredictions.appendChild(li);
    return;
  }

  items.forEach((item) => {
    const li = document.createElement("li");
    li.className = "prediction-chip";
    li.innerHTML = `<strong>${item.digit}</strong><span>${item.confidence.toFixed(2)}%</span>`;
    topPredictions.appendChild(li);
  });
}

function showResult(data) {
  digitResult.textContent = data.digit;
  confidenceResult.textContent =
    data.confidence === null || data.confidence === undefined
      ? "Confidence is unavailable for this saved model."
      : `Confidence: ${data.confidence.toFixed(2)}%`;

  renderPredictions(data.top_predictions || []);
  processedPreview.src = data.processed_preview;

  resultCard.classList.remove("hidden");
  hideError();
}

function clearCanvas() {
  context.clearRect(0, 0, canvas.width, canvas.height);
  fillCanvasBackground();
  state.hasInk = false;
  hideError();
  resultCard.classList.add("hidden");
  processedPreview.removeAttribute("src");
}

async function predictDigit() {
  if (!window.APP_CONFIG.modelReady) {
    showError("The model is not ready yet. Please fix the startup issue shown on the page.");
    return;
  }

  if (!state.hasInk) {
    showError("Draw a digit on the canvas before requesting a prediction.");
    return;
  }

  setLoading(true);
  hideError();

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        image: canvas.toDataURL("image/png"),
      }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    showResult(data);
  } catch (error) {
    showError(error.message || "Unable to connect to the prediction service.");
  } finally {
    setLoading(false);
  }
}

configureCanvas();

canvas.addEventListener("pointerdown", startDrawing);
canvas.addEventListener("pointermove", draw);
canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointerleave", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);

predictButton.addEventListener("click", predictDigit);
clearButton.addEventListener("click", clearCanvas);
