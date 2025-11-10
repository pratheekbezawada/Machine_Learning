let session = null;


async function loadModel() {
  try {
    console.log("🔄 Loading ONNX model...");
    session = await ort.InferenceSession.create("./heart_DLNet_model.onnx");

    console.log("✅ Model Loaded Successfully!");
    const r = document.getElementById("result");
    r.innerText = "✅ Model Loaded!";
    r.style.background = "#d4edda";

  } catch (err) {
    console.error("❌ Model load failed:", err);
    const r = document.getElementById("result");
    r.innerText = "❌ Model Load Failed!";
    r.style.background = "#f8d7da";
  }
}


function toggle(id) {
  const element = document.getElementById(id);
  element.style.display = (element.style.display === "block") ? "none" : "block";
}


function collectInputs() {
  const ids = [
    "Age","Sex","ChestPainType","RestingBP","Cholesterol",
    "FastingBS","RestingECG","MaxHR","ExerciseAngina",
    "Oldpeak","ST_Slope"
  ];
  return new Float32Array(ids.map(id => parseFloat(document.getElementById(id).value) || 0));
}


async function runHeartPrediction() {
  if (!session) {
    alert("Model not loaded yet!");
    return;
  }

  const input = collectInputs();
  const tensor = new ort.Tensor('float32', input, [1, 11]);

  try {
    const outputs = await session.run({input1: tensor});
    const logits = outputs.output1.data;

 
    const e = logits.map(v => Math.exp(v));
    const sum = e.reduce((a,b)=>a+b,0);
    const probs = e.map(v => v / sum);

    const predicted = probs[1] > probs[0] ? 1 : 0;
    const confidence = Math.max(...probs) * 100;

    const result = document.getElementById("result");
    const bar = document.getElementById("barFill");

    if (predicted === 1) {
      result.innerText = "⚠️ High Risk: Heart Disease";
      result.style.background = "#f8d7da";
    } else {
      result.innerText = "✅ Low Risk: No Heart Disease";
      result.style.background = "#d4edda";
    }

    bar.style.width = confidence.toFixed(1) + "%";
    document.getElementById("confidence").innerText =
      "Confidence: " + confidence.toFixed(1) + "%";

  } catch (err) {
    console.error("Prediction Error:", err);
    alert("Prediction Error: " + err.message);
  }
}

window.onload = loadModel;
