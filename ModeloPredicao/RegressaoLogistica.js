const csv = require('csvtojson');
const math = require('mathjs');

async function loadCSV(filePath) {
    const jsonArray = await csv().fromFile(filePath);
    return jsonArray;
}

// Função softmax para calcular probabilidades de classes múltiplas
function softmax(z) {
    const maxZ = Math.max(...z); // Evita overflow subtraindo o maior valor
    const expZ = z.map(val => Math.exp(val - maxZ));
    const sumExpZ = expZ.reduce((acc, val) => acc + val, 0);
    return expZ.map(val => val / sumExpZ);
}

// Inicializar pesos para os atributos
function initializeWeights(attributes) {
    let weights = {};
    attributes.forEach(attr => {
        weights[attr] = {
            H: Math.random(), // Peso para vitória do time da casa
            D: Math.random(), // Peso para empate
            A: Math.random()  // Peso para derrota do time da casa
        };
    });
    return weights;
}

// Treinar modelo de regressão logística multinomial com Gradiente Descendente
function trainLogisticRegression(data, learningRate = 0.01, epochs = 1000) {
    let attributes = Object.keys(data[0]).filter(attr => attr !== 'Res');
    let weights = initializeWeights(attributes);
    let biases = { H: 0, D: 0, A: 0 }; // Biases para cada classe

    for (let epoch = 0; epoch < epochs; epoch++) {
        let totalError = 0;

        data.forEach(row => {
            let x = attributes.map(attr => parseFloat(row[attr]) || 0);
            let y = { H: 0, D: 0, A: 0 };

            // Codificar a classe como one-hot
            y[row['Res']] = 1;

            // Calcular scores para cada classe
            let z = {
                H: biases.H + x.reduce((sum, value, index) => sum + value * weights[attributes[index]].H, 0),
                D: biases.D + x.reduce((sum, value, index) => sum + value * weights[attributes[index]].D, 0),
                A: biases.A + x.reduce((sum, value, index) => sum + value * weights[attributes[index]].A, 0)
            };

            // Calcular probabilidades usando softmax
            let probs = softmax([z.H, z.D, z.A]);

            // Erro entre predição e valor real para cada classe
            let errorH = probs[0] - y.H;
            let errorD = probs[1] - y.D;
            let errorA = probs[2] - y.A;

            totalError += Math.abs(errorH) + Math.abs(errorD) + Math.abs(errorA);

            // Atualizar pesos e biases para cada classe
            for (let i = 0; i < x.length; i++) {
                weights[attributes[i]].H -= learningRate * errorH * x[i];
                weights[attributes[i]].D -= learningRate * errorD * x[i];
                weights[attributes[i]].A -= learningRate * errorA * x[i];
            }
            biases.H -= learningRate * errorH;
            biases.D -= learningRate * errorD;
            biases.A -= learningRate * errorA;
        });

    }

    return { weights, biases };
}

// Fazer uma predição
function predictLogisticRegression(model, sample) {
    let { weights, biases } = model;
    let attributes = Object.keys(weights);
    let x = attributes.map(attr => parseFloat(sample[attr]) || 0);

    // Calcular scores para cada classe
    let z = {
        H: biases.H + x.reduce((sum, value, index) => sum + value * weights[attributes[index]].H, 0),
        D: biases.D + x.reduce((sum, value, index) => sum + value * weights[attributes[index]].D, 0),
        A: biases.A + x.reduce((sum, value, index) => sum + value * weights[attributes[index]].A, 0)
    };

    // Calcular probabilidades usando softmax
    let probs = softmax([z.H, z.D, z.A]);

    // Obter a classe com a maior probabilidade
    let classes = ['H', 'D', 'A'];
    let maxIndex = probs.indexOf(Math.max(...probs));
    return classes[maxIndex];
}

async function runLogisticRegression() {
    let filePath = './data/BRA5anos.csv';
    let data = await loadCSV(filePath);

    let model = trainLogisticRegression(data);

    let sample = {
        'Country': 'Brazil',
        'League': 'Serie A',
        'Season': '2023',
        'Date': '15/10/2023',
        'Time': '16:00',
        'Home': 'Corinthians',
        'Away': 'Flamengo',
        'HG': 2,
        'AG': 10,
        'PH': 2.80,
        'PD': 2.90,
        'PA': 3.00,
        'MaxH': 2.85,
        'MaxD': 2.95,
        'MaxA': 3.10,
        'AvgH': 2.82,
        'AvgD': 2.88,
        'AvgA': 3.05
    };

    let prediction = predictLogisticRegression(model, sample);

    const exibeVencedor = (prediction, sample) => {
        switch (prediction) {
            case 'H':
                return sample.Home;
            case 'A':
                return sample.Away;
            case 'D':
                return 'Empate';
            default:
                return 'Desconhecido';
        }
    };

    console.log(`Predição: ${prediction} | Quem venceu: ${exibeVencedor(prediction, sample)}`);
}

runLogisticRegression();
