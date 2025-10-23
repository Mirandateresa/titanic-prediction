from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ========== CONFIGURACIÓN INICIAL ==========
app = Flask(__name__)
CORS(app)  # Permite peticiones desde cualquier origen

# Configurar la API
api = Api(
    app, 
    version='1.0', 
    title='Titanic Survival Prediction API',
    description='API para predecir la supervivencia en el Titanic usando Machine Learning',
    doc='/docs'  # Habilitar documentación automática en /docs
)

# Namespace para organizar los endpoints
ns = api.namespace('titanic', description='Operaciones de predicción del Titanic')

# ========== MODELOS DE DATOS PARA LA API ==========
passenger_model = api.model('Passenger', {
    'name': fields.String(required=False, description='Nombre completo'),
    'pclass': fields.Integer(required=True, description='Clase (1, 2, 3)', min=1, max=3),
    'sex': fields.String(required=True, description='Sexo (male/female)'),
    'age': fields.Float(required=True, description='Edad', min=0),
    'sibsp': fields.Integer(required=True, description='Número de hermanos/esposos', min=0),
    'parch': fields.Integer(required=True, description='Número de padres/hijos', min=0),
    'fare': fields.Float(required=True, description='Tarifa pagada', min=0),
    'embarked': fields.String(required=True, description='Puerto de embarque (C, Q, S)')
})

# ========== FUNCIONES DE MACHINE LEARNING ==========
def preprocess_data(df):
    """Preprocesa los datos para el modelo"""
    data = df.copy()
    
    # Llenar valores faltantes
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    
    # Extraer título del nombre
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    
    # Clasificar títulos
    title_mapping = {
        'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
        'Dr': 'Professional', 'Rev': 'Professional', 'Col': 'Military',
        'Major': 'Military', 'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Lady': 'Nobility', 'Countess': 'Nobility', 'Sir': 'Nobility',
        'Don': 'Nobility', 'Jonkheer': 'Nobility', 'Dona': 'Nobility',
        'Capt': 'Military'
    }
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'].fillna('Other', inplace=True)
    
    # Crear características adicionales
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    data['IsChild'] = (data['Age'] < 18).astype(int)
    data['IsElderly'] = (data['Age'] > 60).astype(int)
    data['FarePerPerson'] = data['Fare'] / data['FamilySize']
    
    # Clasificar por tarifa
    data['FareCategory'] = pd.cut(data['Fare'], 
                                bins=[0, 25, 100, 600], 
                                labels=['Low', 'Medium', 'High'])
    
    # Codificar variables categóricas
    label_encoders = {}
    categorical_cols = ['Sex', 'Embarked', 'Title', 'FareCategory']
    
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        label_encoders[col] = le
    
    # Seleccionar características para el modelo
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
               'Embarked', 'FamilySize', 'IsAlone', 'IsChild', 
               'IsElderly', 'Title', 'FarePerPerson', 'FareCategory']
    
    return data[features], data['Survived'], label_encoders

def load_or_train_model():
    """Carga el modelo si existe, sino lo entrena"""
    model_path = 'titanic_model.joblib'
    encoders_path = 'label_encoders.joblib'
    
    if os.path.exists(model_path) and os.path.exists(encoders_path):
        # Cargar modelo existente
        model = joblib.load(model_path)
        label_encoders = joblib.load(encoders_path)
        print("✅ Modelo cargado desde archivo")
        return model, label_encoders, None
    else:
        # Entrenar nuevo modelo
        print("🔄 Entrenando nuevo modelo...")
        df = pd.read_csv('titanic.csv')
        X, y, label_encoders = preprocess_data(df)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Entrenar modelo
        model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        # Evaluar modelo
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Guardar modelo
        joblib.dump(model, model_path)
        joblib.dump(label_encoders, encoders_path)
        
        print(f"✅ Modelo entrenado y guardado. Precisión: {accuracy:.2%}")
        return model, label_encoders, accuracy

def preprocess_new_data(new_data, label_encoders):
    """Preprocesa los nuevos datos para predicción"""
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
               'Embarked', 'FamilySize', 'IsAlone', 'IsChild', 
               'IsElderly', 'Title', 'FarePerPerson', 'FareCategory']
    
    processed_data = {}
    
    # Procesar características básicas
    for feature in ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']:
        if feature in new_data:
            processed_data[feature] = new_data[feature]
    
    # Procesar sexo
    if 'sex' in new_data:
        sex_map = {'Masculino': 'male', 'Femenino': 'female', 'male': 'male', 'female': 'female'}
        processed_data['Sex'] = sex_map.get(new_data['sex'], 'male')
    
    # Procesar puerto de embarque
    if 'embarked' in new_data:
        embarked_map = {'C': 'C', 'Q': 'Q', 'S': 'S', 'Cherbourg': 'C', 
                       'Queenstown': 'Q', 'Southampton': 'S'}
        processed_data['Embarked'] = embarked_map.get(new_data['embarked'], 'S')
    
    # Calcular características derivadas
    family_size = new_data.get('sibsp', 0) + new_data.get('parch', 0) + 1
    processed_data['FamilySize'] = family_size
    processed_data['IsAlone'] = 1 if family_size == 1 else 0
    processed_data['IsChild'] = 1 if new_data.get('age', 0) < 18 else 0
    processed_data['IsElderly'] = 1 if new_data.get('age', 0) > 60 else 0
    processed_data['FarePerPerson'] = new_data.get('fare', 0) / family_size
    
    # Categorizar tarifa
    fare = new_data.get('fare', 0)
    if fare <= 25:
        processed_data['FareCategory'] = 'Low'
    elif fare <= 100:
        processed_data['FareCategory'] = 'Medium'
    else:
        processed_data['FareCategory'] = 'High'
    
    # Procesar título
    if 'title' in new_data:
        processed_data['Title'] = new_data['title']
    elif 'name' in new_data:
        # Extraer título del nombre
        name_parts = new_data['name'].split(', ')
        if len(name_parts) > 1:
            title_part = name_parts[1].split('.')[0]
            title_map = {
                'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
                'Dr': 'Professional', 'Rev': 'Professional', 'Col': 'Military',
                'Major': 'Military', 'Lady': 'Nobility', 'Countess': 'Nobility',
                'Sir': 'Nobility', 'Don': 'Nobility', 'Capt': 'Military'
            }
            processed_data['Title'] = title_map.get(title_part, 'Other')
        else:
            processed_data['Title'] = 'Mr'
    else:
        processed_data['Title'] = 'Mr'
    
    # Codificar variables categóricas
    for col in ['Sex', 'Embarked', 'Title', 'FareCategory']:
        if col in processed_data and processed_data[col] in label_encoders[col].classes_:
            processed_data[col] = label_encoders[col].transform([processed_data[col]])[0]
        else:
            # Valor por defecto si no está en el encoder
            processed_data[col] = 0
    
    # Crear DataFrame final
    final_data = pd.DataFrame([processed_data], columns=features)
    
    return final_data

# ========== ENDPOINTS DE LA API ==========
@ns.route('/predict')
class PredictSurvival(Resource):
    @ns.expect(passenger_model)
    @ns.doc('predict_survival',
            responses={
                200: 'Predicción exitosa',
                400: 'Datos inválidos',
                500: 'Error interno del servidor'
            })
    def post(self):
        """Predice si un pasajero hubiera sobrevivido en el Titanic"""
        try:
            # Obtener datos del request
            data = request.json
            
            # Validar datos requeridos
            required_fields = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
            for field in required_fields:
                if field not in data:
                    return {
                        'error': f'Campo requerido faltante: {field}',
                        'required_fields': required_fields
                    }, 400
            
            # Preprocesar datos
            processed_data = preprocess_new_data(data, label_encoders)
            
            # Hacer predicción
            prediction = model.predict(processed_data)[0]
            probability = model.predict_proba(processed_data)[0]
            
            # Preparar respuesta
            result = {
                'survived': bool(prediction),
                'probability_survived': float(probability[1]),
                'probability_died': float(probability[0]),
                'confidence': float(max(probability)),
                'passenger_data': {
                    'name': data.get('name', 'No proporcionado'),
                    'pclass': data['pclass'],
                    'sex': data['sex'],
                    'age': data['age'],
                    'family_size': data['sibsp'] + data['parch'] + 1
                }
            }
            
            return result, 200
            
        except Exception as e:
            return {
                'error': 'Error procesando la solicitud',
                'details': str(e)
            }, 500

@ns.route('/health')
class HealthCheck(Resource):
    @ns.doc('health_check')
    def get(self):
        """Verifica el estado de la API y el modelo"""
        return {
            'status': 'healthy',
            'model_loaded': model is not None,
            'model_accuracy': f"{accuracy:.2%}" if accuracy else "No disponible",
            'message': 'Titanic Survival Prediction API está funcionando correctamente'
        }, 200

@ns.route('/model-info')
class ModelInfo(Resource):
    @ns.doc('model_info')
    def get(self):
        """Obtiene información sobre el modelo"""
        if model is None:
            return {'error': 'Modelo no cargado'}, 500
            
        return {
            'model_type': 'Random Forest',
            'n_estimators': model.n_estimators,
            'features_used': [
                'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                'Embarked', 'FamilySize', 'IsAlone', 'IsChild', 
                'IsElderly', 'Title', 'FarePerPerson', 'FareCategory'
            ],
            'accuracy': f"{accuracy:.2%}" if accuracy else "No disponible"
        }, 200

# ========== INICIALIZACIÓN ==========
print("🚀 Inicializando Titanic Survival Prediction API...")

# Cargar o entrenar modelo al iniciar
model, label_encoders, accuracy = load_or_train_model()

if __name__ == '__main__':
    print("✅ API lista para recibir peticiones!")
    print("📚 Documentación disponible en: http://localhost:5000/docs")
    print("🌐 Health check en: http://localhost:5000/titanic/health")
    app.run(debug=True, host='0.0.0.0', port=5000)