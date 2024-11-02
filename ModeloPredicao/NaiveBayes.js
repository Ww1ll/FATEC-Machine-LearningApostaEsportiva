const csv = require('csvtojson');
const math = require('mathjs'); // Utilizado para funções estatísticas como média e desvio padrão

async function loadCSV(filePath) {
    const jsonArray = await csv().fromFile(filePath);
    return jsonArray;
}

function trainNaiveBayesImproved(data) {
    let classes = {};
    let totalRows = data.length;

    data.forEach(row => {
        let result = row['Res'];
        if (!classes[result]) {
            classes[result] = { total: 0, attributes: {}, means: {}, stds: {} };
        }
        classes[result].total += 1;

        for (let attribute in row) {
            if (attribute === 'Res') continue;

            if (['HG', 'AG', 'PH', 'PD', 'PA', 'MaxH', 'MaxD', 'MaxA', 'AvgH', 'AvgD', 'AvgA'].includes(attribute)) {
                if (!classes[result].attributes[attribute]) {
                    classes[result].attributes[attribute] = [];
                }
                classes[result].attributes[attribute].push(parseFloat(row[attribute]));
            } else {
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

    // Calcular probabilidades a priori e estatísticas de variáveis numéricas
    for (let result in classes) {
        classes[result].priorProbability = classes[result].total / totalRows;
        for (let attribute in classes[result].attributes) {
            if (Array.isArray(classes[result].attributes[attribute])) { // Variável numérica
                classes[result].means[attribute] = math.mean(classes[result].attributes[attribute]);
                classes[result].stds[attribute] = math.std(classes[result].attributes[attribute]);
            } else { // Variável categórica
                for (let value in classes[result].attributes[attribute]) {
                    classes[result].attributes[attribute][value] /= classes[result].total;
                }
            }
        }
    }

    return classes;
}

function gaussianProbability(x, mean, std) {
    return (1 / (Math.sqrt(2 * Math.PI) * std)) * Math.exp(-((x - mean) ** 2) / (2 * std ** 2));
}

function predictNaiveBayesImproved(model, sample) {
    let maxProbability = -1;
    let bestClass = null;

    for (let result in model) {
        let probability = model[result].priorProbability;

        for (let attribute in sample) {
            if (model[result].means[attribute] !== undefined) { // Variável numérica
                let mean = model[result].means[attribute];
                let std = model[result].stds[attribute];
                let value = parseFloat(sample[attribute]);
                let prob = gaussianProbability(value, mean, std);
                probability *= prob;
            } else if (model[result].attributes[attribute] && model[result].attributes[attribute][sample[attribute]]) { // Variável categórica
                probability *= model[result].attributes[attribute][sample[attribute]];
            } else {
                probability *= 0.01; // Suavização para valores desconhecidos
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

    let model = trainNaiveBayesImproved(data);

    let sample = {
    'Country': 'Brazil',
    'League': 'Serie A',
    'Season': '2023',
    'Date': '15/10/2023',
    'Time': '16:00',
    'Home': 'Corinthians',
    'Away': 'Flamengo',
    'HG': 2,       
    'AG': 2,       
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

    let prediction = predictNaiveBayesImproved(model, sample);

    const exibeVencedor = (prediction, sample)=> {
        switch (prediction) {
            case 'H':
                return sample.Home
            case 'A':
                return sample.Away
            default:
                return 'Empate'
        }
    }

    console.log(`Predição: ${prediction} | Quem venceu: ${exibeVencedor(prediction, sample)}`);
}

runNaiveBayes();
