const canvas = document.getElementById("pissarra");
const ctx = canvas.getContext("2d");
const resultatText = document.getElementById("resultat");

// Configuració del dibuix
ctx.fillStyle = "white";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 12; // Gruix ideal per a la IA
ctx.lineCap = "round";
ctx.strokeStyle = "black";

let dibuixant = false;

// Esdeveniments del ratolí
canvas.addEventListener("mousedown", () => dibuixant = true);
canvas.addEventListener("mouseup", () => { dibuixant = false; ctx.beginPath(); });
canvas.addEventListener("mousemove", dibuixar);

function dibuixar(e) {
    if (!dibuixant) return;
    const rect = canvas.getBoundingClientRect();
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
}

// Botó Netejar
document.getElementById("clearBtn").onclick = () => {
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resultatText.innerText = "Pissarra neta. Torna a dibuixar!";
};

// --- FUNCIÓ DE LA IA ---
async function predir() {
    resultatText.innerText = "La IA està pensant...";

    try {
        // Carreguem el model (assegura't que el fitxer es diu model.onnx a GitHub)
        const session = await ort.InferenceSession.create('./model.onnx');
        
        // Preparem el dibuix per a la IA (ha de ser 28x28 píxels)
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(canvas, 0, 0, 28, 28);
        
        const imgData = tempCtx.getImageData(0, 0, 28, 28);
        const input = new Float32Array(28 * 28);
        
        for (let i = 0; i < imgData.data.length; i += 4) {
            // Convertim a escala de grisos (0-1) i invertim (el fons ha de ser 0 i el traç 1)
            const avg = (imgData.data[i] + imgData.data[i+1] + imgData.data[i+2]) / 3;
            input[i / 4] = (255 - avg) / 255.0;
        }

        // Creem el tensor d'entrada
        const tensor = new ort.Tensor('float32', input, [1, 1, 28, 28]);
        
        // Executem la predicció
        const output = await session.run({ [session.inputNames[0]]: tensor });
        const data = output[session.outputNames[0]].data;

        // Resultats
        const classes = ["Cercle", "Quadrat", "Triangle"];
        const maxIdx = data.indexOf(Math.max(...data));
        
        resultatText.innerText = "Predicció: " + classes[maxIdx];

    } catch (e) {
        console.error(e);
        resultatText.innerText = "Error: " + e.message;
    }
}

document.getElementById("botoPredir").onclick = predir;
