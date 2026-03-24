// Esperem que la finestra estigui carregada
window.onload = function() {
    const canvas = document.getElementById("pissarra");
    const ctx = canvas.getContext("2d");
    const resultatText = document.getElementById("resultat");

    // 1. CONFIGURACIÓ DEL DIBUIX
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 15; // Gruix per facilitar la feina a la IA
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";

    let dibuixant = false;

    canvas.addEventListener("mousedown", (e) => { dibuixant = true; dibuixar(e); });
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
        ctx.fillStyle = "white";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        resultatText.innerText = "Pissarra neta";
    };

    // 2. LÒGICA DE LA IA
    document.getElementById("botoPredir").onclick = async () => {
        resultatText.innerText = "La IA està pensant...";

        try {
            // Carreguem el model (assegura't que el fitxer es digui model.onnx a GitHub)
            // window.ort és com accedim a la llibreria sense usar 'import'
            const session = await window.ort.InferenceSession.create('./model.onnx');
            
            // Creem un canvas petit de 28x28 per processar la imatge
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 28;
            tempCanvas.height = 28;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(canvas, 0, 0, 28, 28);
            
            const imgData = tempCtx.getImageData(0, 0, 28, 28);
            const input = new Float32Array(28 * 28);
            
            for (let i = 0; i < imgData.data.length; i += 4) {
                // Convertim a grisos i normalitzem (0 a 1)
                const avg = (imgData.data[i] + imgData.data[i+1] + imgData.data[i+2]) / 3;
                input[i / 4] = (255 - avg) / 255.0; // Fons blanc esdevé 0, traç negre esdevé 1
            }

            const tensor = new window.ort.Tensor('float32', input, [1, 1, 28, 28]);
            const output = await session.run({ [session.inputNames[0]]: tensor });
            const data = output[session.outputNames[0]].data;

            const classes = ["Cercle", "Quadrat", "Triangle"];
            const maxIdx = data.indexOf(Math.max(...data));
            
            resultatText.innerText = "Predicció: " + classes[maxIdx];

        } catch (e) {
            console.error(e);
            resultatText.innerText = "Error: No s'ha pogut carregar el model. Revisa el fitxer .onnx";
        }
    };
};
