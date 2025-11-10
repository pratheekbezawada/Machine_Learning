let session = null;
let inputName = null;
let outputName = null;

function $(id) { return document.getElementById(id); }

function getInputsAsFloat32() {
    const ids = [
        "Trip_Distance_km",
        "Time",
        "Day",
        "Passenger_Count",
        "Traffic_Conditions",
        "Weather",
        "Base_Fare",
        "Per_Km_Rate",
        "Per_Minute_Rate",
        "Trip_Duration_Minutes"
    ];
    return new Float32Array(ids.map(id => parseFloat($(id).value) || 0));
}

async function loadModel() {
    try {
        session = await ort.InferenceSession.create("./xgboost_TaxiData.onnx");
        inputName = session.inputNames?.[0] || "input1";
        outputName = session.outputNames?.[0] || "output1";
    } catch (err) {
        alert("Model Load Failed!");
        console.error(err);
    }
}

async function runPricePrediction() {
    if (!session) return alert("Model not loaded!");

    const inputArray = getInputsAsFloat32();
    const tensor = new ort.Tensor("float32", inputArray, [1, 10]);

    try {
        const output = await session.run({ [inputName]: tensor });
        const out = output[outputName] || output[Object.keys(output)[0]];
        const predicted = Array.from(out.data)[0].toFixed(2);

        const r = $("result");
        r.style.display = "block";
        r.innerText = "Predicted Price: ₹" + predicted;
    } catch (e) {
        $("result").style.display = "block";
        $("result").innerText = "Prediction Error";
        console.error(e);
    }
}

window.onload = loadModel;
