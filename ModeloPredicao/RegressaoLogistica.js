const csv = require('csvtojson');
const math = require('mathjs');

function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
}

async function loadCSV(filePath) {
    const jsonArray = await csv().fromFile(filePath);
    return jsonArray;
}

function trainLogisticRegression(data, learningRate = 0.01, iterations = 1000) {
    let weights = {};
    let totalRows = data.length;
    let features = Object.keys(data[0]).filter(attr => attr !== 'Res');

    features.forEach(attr => {
        weights[attr] = Math.random();
    });

    for (let i = 0; i < iterations; i++) {
        let gradients = {};
        features.forEach(attr => gradients[attr] = 0);

        data.forEach(row => {
            let z = 0;
            features.forEach(attr => {
                z += weights[attr] * parseFloat(row[attr]);
            });

            let prediction = sigmoid(z);
            let error = prediction - (row['Res'] === 'H' ? 1 : 0);

            features.forEach(attr => {
                gradients[attr] += error * parseFloat(row[attr]);
            });
        });

        features.forEach(attr => {
            weights[attr] -= (learningRate / totalRows) * gradients[attr];
        });
    }

    return weights;
}

function predictLogisticRegression(weights, sample) {
    let z = 0;
    Object.keys(sample).forEach(attr => {
        if (weights[attr]) {
            z += weights[attr] * parseFloat(sample[attr]);
        }
    });
    return sigmoid(z) >= 0.5 ? 'H' : 'A';
}

async function runLogisticRegression() {
    let filePath = './data/BRA5anos.csv';
    let data = await loadCSV(filePath);

    let weights = trainLogisticRegression(data);

    let sample = {
        'Country': 'Brazil',
        'League': 'Serie A',
        'Season': '2023',
        'Date': '09/10/2023',
        'Time': '18:00',
        'Home': 'Palmeiras',
        'Away': 'Fluminense',
        'HG': 3,
        'AG': 1,
        'PH': 1.85,
        'PD': 3.60,
        'PA': 4.10,
        'MaxH': 1.90,
        'MaxD': 3.70,
        'MaxA': 4.20,
        'AvgH': 1.82,
        'AvgD': 3.55,
        'AvgA': 4.00
    };

    let prediction = predictLogisticRegression(weights, sample);
    console.log(`Predição: ${prediction} | Time: ${sample.Home}`);
}

runLogisticRegression();