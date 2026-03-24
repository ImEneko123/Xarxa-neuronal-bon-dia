import rutaModel from "./model.onnx";
// 1r TRUC: Enganyem el CodeSandbox perquè no busqui rutes de servidor
window.__dirname = "/";

// ==========================================
// 0. CARREGAR LA IA MANUALMENT (VERSIÓ ESTABLE 1.14.0)
// ==========================================
let iaCarregada = false;

const scriptIA = document.createElement("script");
// Baixem a la versió 1.14.0, que és molt més robusta i compatible amb CodeSandbox
scriptIA.src =
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/ort.min.js";

scriptIA.onload = () => {
  window.ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/";

  // 2n TRUC: Evitem l'error groc dels "Threads"
  window.ort.env.wasm.numThreads = 1;

  iaCarregada = true;
  document.getElementById("resultat").innerText =
    "Cervell connectat! Ja pots dibuixar.";
};
document.head.appendChild(scriptIA);

// ==========================================
// 1. CONFIGURACIÓ DE LA PISSARRA I EL DIBUIX
// ==========================================
const canvas = document.getElementById("pissarra");
const ctx = canvas.getContext("2d");

ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);

ctx.lineWidth = 1.5;
ctx.lineCap = "round";
ctx.strokeStyle = "black";

let dibuixant = false;

function obtenirPosicio(event) {
  const rect = canvas.getBoundingClientRect();
  const escalaX = canvas.width / rect.width;
  const escalaY = canvas.height / rect.height;
  return {
    x: (event.clientX - rect.left) * escalaX,
    y: (event.clientY - rect.top) * escalaY,
  };
}

canvas.addEventListener("mousedown", (e) => {
  dibuixant = true;
  const pos = obtenirPosicio(e);
  ctx.beginPath();
  ctx.moveTo(pos.x, pos.y);
});

canvas.addEventListener("mousemove", (e) => {
  if (!dibuixant) return;
  const pos = obtenirPosicio(e);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
});

canvas.addEventListener("mouseup", () => (dibuixant = false));
canvas.addEventListener("mouseout", () => (dibuixant = false));

// ==========================================
// 2. BOTÓ DE NETEJAR LA PISSARRA
// ==========================================
document.getElementById("clearBtn").addEventListener("click", () => {
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById("resultat").innerText = iaCarregada
    ? "Pissarra neta!"
    : "";
});

// ==========================================
// 3. INTEL·LIGÈNCIA ARTIFICIAL (PREDICCIÓ)
// ==========================================
function preprocessCanvas() {
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = new Float32Array(28 * 28);

  for (let i = 0; i < 28 * 28; i++) {
    const pixelVal = imageData.data[i * 4];
    data[i] = (255 - pixelVal) / 255.0;
  }

  return new window.ort.Tensor("float32", data, [1, 1, 28, 28]);
}

document.getElementById("botoPredir").addEventListener("click", async () => {
  const resultatText = document.getElementById("resultat");

  if (!iaCarregada) {
    resultatText.innerText = "Espera un segon! La IA s'està preparant...";
    return;
  }

  resultatText.innerText = "Pensant...";

  try {
    // 3r TRUC: Descarreguem el model manualment i evitem errors interns
    const respostaDescarrega = await fetch(rutaModel);
    const modelMastegat = await respostaDescarrega.arrayBuffer();

    // Comprovem si l'arxiu està buit o corrupte
    if (modelMastegat.byteLength < 1000) {
      throw new Error(
        "L'arxiu .onnx està trencat o CodeSandbox no el troba. Mida: " +
          modelMastegat.byteLength +
          " bytes."
      );
    }

    // Forcem la IA a funcionar amb la tecnologia més bàsica i estable
    const opcions = { executionProviders: ["wasm"] };
    const session = await window.ort.InferenceSession.create(
      modelMastegat,
      opcions
    );

    const tensor = preprocessCanvas();
    const feeds = { [session.inputNames[0]]: tensor };
    const outputData = await session.run(feeds);
    const output = outputData[session.outputNames[0]].data;

    const classes = ["Cercle", "Quadrat", "Triangle"];
    let maxIndex = 0;
    let maxValue = output[0];

    for (let i = 1; i < output.length; i++) {
      if (output[i] > maxValue) {
        maxValue = output[i];
        maxIndex = i;
      }
    }

    resultatText.innerText = "Crec que és un: " + classes[maxIndex];
  } catch (error) {
    console.error("ERROR EXACTE:", error);
    resultatText.innerText =
      "Ups! Hi ha hagut un error llegint la IA. Mira la consola.";
  }
});
