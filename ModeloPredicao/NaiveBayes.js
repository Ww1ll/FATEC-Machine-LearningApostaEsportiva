const csv = require('csvtojson');

async function loadCSV(filePath) {
    const jsonArray = await csv().fromFile(filePath);
    return jsonArray;
}

function trainNaiveBayes(data) {
    let classes = {};
    let totalRows = data.length;
 
    data.forEach(row => {
        let result = row['Res']; 
        if (!classes[result]) {
            classes[result] = { total: 0, attributes: {} };
        }
        classes[result].total += 1;
 
        for (let attribute in row) {
            if (attribute !== 'Res') {
                if (!classes[result].attributes[attribute]) {
                    classes[result].attributes[attribute] = {};
                }
                let value = row[attribute];
                if (!classes[result].attributes[attribute][value]) {
                    classes[result].attributes[attribute][value] = 0;
                }
                classes[result].attributes[attribute][value] += 1;
            }
        }
    });
 
    for (let result in classes) {
        classes[result].priorProbability = classes[result].total / totalRows;
        for (let attribute in classes[result].attributes) {
            for (let value in classes[result].attributes[attribute]) {
                classes[result].attributes[attribute][value] /= classes[result].total;
            }
        }
    }
 
    return classes;
}
 
function predictNaiveBayes(model, sample) {
    let maxProbability = -1;
    let bestClass = null;
 
    for (let result in model) {
        let probability = model[result].priorProbability;
 
        for (let attribute in sample) {
            if (model[result].attributes[attribute] && model[result].attributes[attribute][sample[attribute]]) {
                probability *= model[result].attributes[attribute][sample[attribute]];
            } else {
                probability *= 0.01; 
            }
        }
 
        if (probability > maxProbability) {
            maxProbability = probability;
            bestClass = result;
        }
    }
 
    return bestClass;
}
 
async function runNaiveBayes() {
    let filePath = './data/BRA5anos.csv'; 
    let data = await loadCSV(filePath);
 
    let model = trainNaiveBayes(data);
 
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
 
    let prediction = predictNaiveBayes(model, sample);
    console.log(`Predição: ${prediction} | Time: ${sample.Home}`);
}
 
runNaiveBayes();