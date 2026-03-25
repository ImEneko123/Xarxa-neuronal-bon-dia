// --- 1. CONFIGURACIÓ DEL LLENÇ (CANVAS) ---
const canvas = document.querySelector('canvas');
const ctx = canvas.getContext('2d');

let dibuixant = false;

// Configurar l'estil del llapis (Gruix 25, perfecte per dibuixar)
ctx.lineWidth = 25; 
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';

// OMPLIM EL FONS DE BLANC (Així la IA no es confon amb el fons transparent)
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Escoltar el ratolí per poder dibuixar
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

// Funció per netejar la pissarra i tornar-la a pintar de blanc
function netejarPissarra() {
    ctx.fillStyle = 'white'; 
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
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
        // Carreguem el fitxer de la IA
        const session = await ort.InferenceSession.create('./model_definitiu.onnx');

        // Creem un canvas petit amagat de 28x28
        const canvasPetit = document.createElement('canvas');
        canvasPetit.width = 28;
        canvasPetit.height = 28;
        const ctxPetit = canvasPetit.getContext('2d');
        
        // Copiem el dibuix gran i el fem petit
        ctxPetit.drawImage(canvas, 0, 0, 28, 28);
        const imageData = ctxPetit.getImageData(0, 0, 28, 28);
        const data = imageData.data;
        
        // Convertim els píxels a l'idioma de la IA
        const inputData = new Float32Array(28 * 28);
        for (let i = 0; i < 28 * 28; i++) {
            const r = data[i * 4]; // Agafem el color del píxel
            
            // FÓRMULA MÀGICA: Converteix fons blanc en 0.0 i traç negre en 1.0
            inputData[i] = 1.0 - (r / 255.0); 
        }

        // Crear el tensor i executar la xarxa
        const tensor = new ort.Tensor('float32', inputData, [1, 1, 28, 28]);
        const feeds = { input: tensor };
        const results = await session.run(feeds);

        // Agafar els resultats
        const output = results.output.data;
        
        // Buscar quina és l'opció guanyadora
        let maxIndex = 0;
        let maxValue = output[0];
        for (let i = 1; i < output.length; i++) {
            if (output[i] > maxValue) {
                maxValue = output[i];
                maxIndex = i;
            }
        }

        // Llista de formes
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
document.getElementById('btn-predir').addEventListener('click', predir);
document.getElementById('btn-netejar').addEventListener('click', netejarPissarra);
