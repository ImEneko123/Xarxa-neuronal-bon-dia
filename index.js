// --- 1. CONFIGURACIÓ DEL LLENÇ (CANVAS) ---
const canvas = document.querySelector('canvas');
const ctx = canvas.getContext('2d');

// Variables per controlar el dibuix
let dibuixant = false;

// Configurar l'estil del llapis
ctx.lineWidth = 1.5; 
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';

// Funcions per dibuixar amb el ratolí
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

// Netejar la pissarra
function netejarPissarra() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    document.getElementById('resultat').innerText = ''; // Neteja també el text d'error o resultat
}

// --- 2. INTEL·LIGÈNCIA ARTIFICIAL (ONNX) ---
async function predir() {
    const textResultat = document.getElementById('resultat');
    textResultat.innerText = 'Pensant...';
    textResultat.style.color = 'blue';

    try {
        // LA LÍNIA MÀGICA: Li passem directament l'enllaç perquè trobi el model.onnx i el model.onnx.data junts!
        const session = await ort.InferenceSession.create('./model.onnx');

        // 1. Agafar la imatge del canvas (assumint que el teu canvas originalment és de 28x28)
        const imageData = ctx.getImageData(0, 0, 28, 28);
        const data = imageData.data;
        
        // 2. Convertir la imatge per a la IA (Blanc i negre, de 0 a 1)
        const inputData = new Float32Array(28 * 28);
        for (let i = 0; i < 28 * 28; i++) {
            // Com que dibuixes en negre sobre fons transparent/blanc, 
            // invertim els colors perquè la IA va ser entrenada amb línies blanques sobre fons negre
            const r = data[i * 4];
            const g = data[i * 4 + 1];
            const b = data[i * 4 + 2];
            const alpha = data[i * 4 + 3];

            // Si hi ha dibuix (alpha > 0), ho marquem com a blanc (1.0), sinó negre (0.0)
            if (alpha > 0 && (r < 100 && g < 100 && b < 100)) {
                inputData[i] = 1.0;
            } else {
                inputData[i] = 0.0;
            }
        }

        // 3. Crear el "Tensor" (el paquet de dades amb la forma que espera Python: 1 imatge, 1 canal, 28x28)
        const tensor = new ort.Tensor('float32', inputData, [1, 1, 28, 28]);

        // 4. Executar la predicció (l'entrada es diu 'input' tal com vas posar al teu Python)
        const feeds = { input: tensor };
        const results = await session.run(feeds);

        // 5. Llegir el resultat (la sortida es diu 'output')
        const output = results.output.data;
        
        // Buscar quina forma té la puntuació més alta
        let maxIndex = 0;
        let maxValue = output[0];
        for (let i = 1; i < output.length; i++) {
            if (output[i] > maxValue) {
                maxValue = output[i];
                maxIndex = i;
            }
        }

        // 6. Mostrar el resultat final
        const formes = ['Cercle 🟢', 'Quadrat 🟦', 'Triangle 🔺'];
        textResultat.innerText = 'És un ' + formes[maxIndex] + '!';
        textResultat.style.color = 'green';

    } catch (error) {
        console.error("Error de la IA:", error);
        textResultat.innerText = "Error: No s'ha pogut executar el model.";
        textResultat.style.color = 'red';
    }
}

// Associar els botons a les funcions (ajusta els IDs segons el teu HTML)
document.querySelector('button[onclick="predir()"]').addEventListener('click', predir);
document.querySelector('button[onclick="netejar()"]').addEventListener('click', netejarPissarra);
