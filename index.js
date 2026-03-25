// --- 1. CONFIGURACIÓ DEL LLENÇ (CANVAS) ---
const canvas = document.querySelector('canvas');
const ctx = canvas.getContext('2d');

let dibuixant = false;

// Configurar l'estil del llapis
ctx.lineWidth = 12; // Gruixut perquè la IA ho vegi clar
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';

// Escoltar el ratolí
canvas.addEventListener('mousedown', (e) => {
    dibuixant = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener('mousemove', (e) => {
    if (dibuixant) {
        ctx.lineTo(e.offsetX, e.offsetY);
        ctx.stroke();
    }
});

canvas.addEventListener('mouseup', () => {
    dibuixant = false;
});

canvas.addEventListener('mouseleave', () => {
    dibuixant = false;
});

// Funció per netejar la pissarra
function netejarPissarra() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const textResultat = document.getElementById('resultat');
    textResultat.innerText = 'Dibuixa alguna cosa per començar';
    textResultat.style.color = 'blue';
}

// --- 2. INTEL·LIGÈNCIA ARTIFICIAL (ONNX) ---
async function predir() {
    const textResultat = document.getElementById('resultat');
    textResultat.innerText = 'Pensant...';
    textResultat.style.color = 'blue';

    try {
        // Carreguem el fitxer bo de la IA
        const session = await ort.InferenceSession.create('./model_definitiu.onnx');

        // Creem un canvas petit amagat de 28x28 per donar-li la mida correcta a la IA
        const canvasPetit = document.createElement('canvas');
        canvasPetit.width = 28;
        canvasPetit.height = 28;
        const ctxPetit = canvasPetit.getContext('2d');
        
        // Copiem el dibuix gran i el fem petit
        ctxPetit.drawImage(canvas, 0, 0, 28, 28);
        const imageData = ctxPetit.getImageData(0, 0, 28, 28);
        const data = imageData.data;
        
        // Convertim els píxels en zeros i uns
        const inputData = new Float32Array(28 * 28);
        for (let i = 0; i < 28 * 28; i++) {
            const r = data[i * 4];
            const g = data[i * 4 + 1];
            const b = data[i * 4 + 2];
            const alpha = data[i * 4 + 3];

            // Invertim els colors (la IA llegeix blanc sobre negre)
            if (alpha > 0 && (r < 100 && g < 100 && b < 100)) {
                inputData[i] = 1.0; 
            } else {
                inputData[i] = 0.0;
            }
        }

        // Crear el tensor amb les dades
        const tensor = new ort.Tensor('float32', inputData, [1, 1, 28, 28]);

        // Executar la xarxa neuronal
        const feeds = { input: tensor };
        const results = await session.run(feeds);

        // Agafar els resultats
        const output = results.output.data;
        
        // Buscar la puntuació més alta
        let maxIndex = 0;
        let maxValue = output[0];
        for (let i = 1; i < output.length; i++) {
            if (output[i] > maxValue) {
                maxValue = output[i];
                maxIndex = i;
            }
        }

        // Llista de formes (Ajusta l'ordre si la teva IA els té diferent a Python!)
        const formes = ['Cercle 🟢', 'Quadrat 🟦', 'Triangle 🔺'];
        textResultat.innerText = 'És un ' + formes[maxIndex] + '!';
        textResultat.style.color = 'green';

    } catch (error) {
        console.error("Error de la IA:", error);
        textResultat.innerText = "S'ha produït un error a l'executar el model.";
        textResultat.style.color = 'red';
    }
}

// --- 3. CONNECTAR ELS BOTONS ---
// Ara els busquem pel seu "id" en lloc d'un nom estrany, així no falla mai!
document.getElementById('btn-predir').addEventListener('click', predir);
document.getElementById('btn-netejar').addEventListener('click', netejarPissarra);
