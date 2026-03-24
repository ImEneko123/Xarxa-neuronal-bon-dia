window.onload = function() {
    const canvas = document.getElementById("pissarra");
    const ctx = canvas.getContext("2d");
    const resultatText = document.getElementById("resultat");

    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 15; 
    ctx.lineCap = "round";
    ctx.strokeStyle = "black";

    let dibuixant = false;

    canvas.onmousedown = (e) => { dibuixant = true; };
    canvas.onmouseup = () => { dibuixant = false; ctx.beginPath(); };
    canvas.onmousemove = (e) => {
        if (!dibuixant) return;
        const rect = canvas.getBoundingClientRect();
        ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
        ctx.stroke();
    };

    document.getElementById("clearBtn").onclick = () => {
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        resultatText.innerText = "Pissarra neta";
    };

    document.getElementById("botoPredir").onclick = async () => {
        resultatText.innerText = "La IA està pensant...";
        try {
            // Hem simplificat la càrrega per evitar l'error 8537680
            const session = await window.ort.InferenceSession.create('./model.onnx', { 
                executionProviders: ['wasm'] 
            });
            
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = 28; tempCanvas.height = 28;
            tempCanvas.getContext('2d').drawImage(canvas, 0, 0, 28, 28);
            const imgData = tempCanvas.getContext('2d').getImageData(0, 0, 28, 28);
            
            const input = new Float32Array(28 * 28);
            for (let i = 0; i < imgData.data.length; i += 4) {
                input[i / 4] = (255 - imgData.data[i]) / 255.0;
            }

            const tensor = new window.ort.Tensor('float32', input, [1, 1, 28, 28]);
            const output = await session.run({ [session.inputNames[0]]: tensor });
            const data = output[session.outputNames[0]].data;

            const classes = ["Cercle", "Quadrat", "Triangle"];
            const maxIdx = data.indexOf(Math.max(...data));
            resultatText.innerText = "Predicció: " + classes[maxIdx];

        } catch (e) {
            console.error(e);
            resultatText.innerText = "Error: No es pot carregar el fitxer model.onnx";
        }
    };
};
