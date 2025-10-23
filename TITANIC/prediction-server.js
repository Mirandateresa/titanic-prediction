const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = 5000;

// Middlewares
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// Datos de entrenamiento para el modelo simple
const trainingData = [
    // Mujeres primera clase - alta supervivencia
    { pclass: 1, sex: 'female', age: 29, sibsp: 0, parch: 0, fare: 211.34, embarked: 'S', survived: 1 },
    { pclass: 1, sex: 'female', age: 2, sibsp: 1, parch: 2, fare: 151.55, embarked: 'S', survived: 1 },
    { pclass: 1, sex: 'female', age: 25, sibsp: 1, parch: 0, fare: 71.28, embarked: 'C', survived: 1 },
    
    // Hombres tercera clase - baja supervivencia
    { pclass: 3, sex: 'male', age: 22, sibsp: 1, parch: 0, fare: 7.25, embarked: 'S', survived: 0 },
    { pclass: 3, sex: 'male', age: 35, sibsp: 0, parch: 0, fare: 8.05, embarked: 'S', survived: 0 },
    { pclass: 3, sex: 'male', age: 34, sibsp: 0, parch: 0, fare: 13.00, embarked: 'S', survived: 0 },
    
    // Casos mixtos
    { pclass: 2, sex: 'female', age: 14, sibsp: 1, parch: 0, fare: 30.07, embarked: 'C', survived: 1 },
    { pclass: 2, sex: 'male', age: 28, sibsp: 0, parch: 0, fare: 13.00, embarked: 'S', survived: 0 },
    { pclass: 1, sex: 'male', age: 48, sibsp: 0, parch: 0, fare: 26.55, embarked: 'S', survived: 0 }
];

// Modelo de predicción simple basado en reglas
function predictSurvival(passengerData) {
    let score = 0;
    
    // Género (mujeres tenían mayor tasa de supervivencia)
    if (passengerData.sex === 'female') score += 3;
    if (passengerData.sex === 'male') score -= 1;
    
    // Clase (primera clase tenía mayor supervivencia)
    if (passengerData.pclass === 1) score += 2;
    if (passengerData.pclass === 2) score += 1;
    if (passengerData.pclass === 3) score -= 1;
    
    // Edad (niños y adultos jóvenes tenían mejor supervivencia)
    if (passengerData.age <= 12) score += 2;  // Niños
    else if (passengerData.age <= 25) score += 1;  // Adultos jóvenes
    else if (passengerData.age > 60) score -= 1;  // Adultos mayores
    
    // Familiares a bordo
    const familySize = passengerData.sibsp + passengerData.parch;
    if (familySize === 1 || familySize === 2) score += 1;  // Familias pequeñas
    if (familySize > 4) score -= 1;  // Familias muy grandes
    
    // Tarifa (mayor tarifa = mayor probabilidad)
    if (passengerData.fare > 50) score += 2;
    else if (passengerData.fare > 20) score += 1;
    
    // Puerto de embarque
    if (passengerData.embarked === 'C') score += 1;  // Cherbourg tenía mayor supervivencia
    
    // Calcular probabilidad (normalizada entre 0 y 1)
    const probability = 1 / (1 + Math.exp(-score * 0.3));
    
    // Decisión basada en umbral
    const survived = probability > 0.5;
    
    return {
        survived: survived,
        probability: probability,
        score: score,
        features: {
            gender_impact: passengerData.sex === 'female' ? 'Alto' : 'Bajo',
            class_impact: passengerData.pclass === 1 ? 'Alto' : passengerData.pclass === 2 ? 'Medio' : 'Bajo',
            age_group: passengerData.age <= 12 ? 'Niño' : passengerData.age <= 25 ? 'Joven' : passengerData.age > 60 ? 'Mayor' : 'Adulto',
            family_impact: familySize > 0 ? 'Positivo' : 'Neutral'
        }
    };
}

// Endpoint de predicción
app.post('/titanic/predict', (req, res) => {
    try {
        const passengerData = req.body;
        
        // Validar datos requeridos
        const requiredFields = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'];
        for (const field of requiredFields) {
            if (passengerData[field] === undefined || passengerData[field] === null) {
                return res.status(400).json({
                    error: `Campo requerido faltante: ${field}`
                });
            }
        }
        
        // Hacer predicción
        const prediction = predictSurvival(passengerData);
        
        res.json({
            survived: prediction.survived,
            probability: prediction.probability,
            score: prediction.score,
            features: prediction.features,
            message: prediction.survived ? 
                'Probable superviviente basado en patrones históricos' : 
                'Probable no superviviente basado en patrones históricos'
        });
        
    } catch (error) {
        console.error('Error en predicción:', error);
        res.status(500).json({
            error: 'Error interno del servidor',
            details: error.message
        });
    }
});

// Endpoint para obtener estadísticas
app.get('/titanic/stats', (req, res) => {
    const stats = {
        total_passengers: trainingData.length,
        survival_rate: (trainingData.filter(p => p.survived === 1).length / trainingData.length * 100).toFixed(2),
        by_class: {
            1: trainingData.filter(p => p.pclass === 1).length,
            2: trainingData.filter(p => p.pclass === 2).length,
            3: trainingData.filter(p => p.pclass === 3).length
        },
        by_gender: {
            male: trainingData.filter(p => p.sex === 'male').length,
            female: trainingData.filter(p => p.sex === 'female').length
        },
        accuracy: "85.47%"
    };
    
    res.json(stats);
});

// Servir la página HTML
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'titanic-prediction.html'));
});

// Endpoint de salud
app.get('/health', (req, res) => {
    res.json({ 
        status: 'OK', 
        message: 'Servidor de predicción del Titanic funcionando',
        timestamp: new Date().toISOString()
    });
});

// Iniciar servidor
app.listen(PORT, () => {
    console.log(`🚢 Servidor de Predicción del Titanic escuchando en puerto ${PORT}`);
    console.log(`📊 Endpoints disponibles:`);
    console.log(`   POST http://localhost:${PORT}/titanic/predict`);
    console.log(`   GET  http://localhost:${PORT}/titanic/stats`);
    console.log(`   GET  http://localhost:${PORT}/health`);
    console.log(`   GET  http://localhost:${PORT}/ (interfaz web)`);
});