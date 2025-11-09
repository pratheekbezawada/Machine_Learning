import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js";

async function loadModel() {
  try {
    const session = await ort.InferenceSession.create("./heart_mlp_model.onnx");
    console.log("✅ Model loaded successfully");
    return session;
  } catch (err) {
    console.error("Model load error:", err);
    document.getElementById("result").textContent = "❌ Failed to load model.";
  }
}

function getInputValues() {
  const fields = [
    "Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS",
    "RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"
  ];
  return new Float32Array(fields.map(id => parseFloat(document.getElementById(id).value) || 0));
}

async function runPrediction(session) {
  const input = getInputValues();
  const tensor = new ort.Tensor("float32", input, [1, 11]);
  const feeds = { input1: tensor };
  const output = await session.run(feeds);
  const logits = output.output1.data;

  // Apply softmax manually
  const exp = logits.map(x => Math.exp(x));
  const sumExp = exp.reduce((a,b) => a + b, 0);
  const probs = exp.map(x => x / sumExp);
  const pred = probs[1] > probs[0] ? 1 : 0;
  const confidence = Math.max(...probs);

  // Update UI
  const resultDiv = document.getElementById("result");
  const bar = document.getElementById("barFill");
  const confText = document.getElementById("confidence");

  if (pred === 1) {
    resultDiv.textContent = `⚠️ Predicted: Heart Disease (1)`;
    resultDiv.style.background = "#ffe5e5";
  } else {
    resultDiv.textContent = `✅ Predicted: No Disease (0)`;
    resultDiv.style.background = "#e5ffe5";
  }

  const percent = (confidence * 100).toFixed(1);
  bar.style.width = `${percent}%`;
  confText.textContent = `Confidence: ${percent}%`;
}

// Initialize
const session = await loadModel();
document.getElementById("predictBtn").addEventListener("click", () => runPrediction(session));
