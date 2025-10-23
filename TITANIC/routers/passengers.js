const express = require('express');
const router = express.Router();
const fs = require('fs');
const path = require('path');

// Cargar datos desde el archivo JSON
const dataPath = path.join(__dirname, '../data/titanic.json');
let passengersData;

try {
  passengersData = JSON.parse(fs.readFileSync(dataPath, 'utf8'));
  console.log(`✅ Datos cargados: ${passengersData.passengers.length} pasajeros`);
} catch (error) {
  console.error('❌ Error cargando datos:', error.message);
  // Datos de respaldo si el archivo no existe
  passengersData = { passengers: [] };
}

// GET - Todos los pasajeros
router.get('/', (req, res) => {
  try {
    const { page = 1, limit = 20 } = req.query;
    const startIndex = (page - 1) * limit;
    const endIndex = page * limit;

    const passengers = passengersData.passengers.slice(startIndex, endIndex);
    
    res.json({
      total: passengersData.passengers.length,
      page: parseInt(page),
      limit: parseInt(limit),
      data: passengers
    });
  } catch (error) {
    res.status(500).json({ error: 'Error al obtener pasajeros' });
  }
});

// GET - Pasajero por ID
router.get('/:id', (req, res) => {
  try {
    const passenger = passengersData.passengers.find(
      p => p.PassengerId === parseInt(req.params.id)
    );
    
    if (!passenger) {
      return res.status(404).json({ error: 'Pasajero no encontrado' });
    }
    
    res.json(passenger);
  } catch (error) {
    res.status(500).json({ error: 'Error al buscar pasajero' });
  }
});

// GET - Filtrar por clase
router.get('/class/:class', (req, res) => {
  try {
    const pclass = parseInt(req.params.class);
    const passengers = passengersData.passengers.filter(
      p => p.Pclass === pclass
    );
    
    res.json({
      class: pclass,
      count: passengers.length,
      data: passengers
    });
  } catch (error) {
    res.status(500).json({ error: 'Error al filtrar por clase' });
  }
});

// GET - Filtrar por supervivencia
router.get('/survived/:status', (req, res) => {
  try {
    const survived = parseInt(req.params.status);
    const passengers = passengersData.passengers.filter(
      p => p.Survived === survived
    );
    
    res.json({
      survived: survived === 1 ? 'Sobrevivió' : 'No sobrevivió',
      count: passengers.length,
      data: passengers
    });
  } catch (error) {
    res.status(500).json({ error: 'Error al filtrar por supervivencia' });
  }
});

// GET - Buscar por nombre
router.get('/search/:name', (req, res) => {
  try {
    const searchTerm = req.params.name.toLowerCase();
    const passengers = passengersData.passengers.filter(
      p => p.Name.toLowerCase().includes(searchTerm)
    );
    
    res.json({
      search: searchTerm,
      count: passengers.length,
      data: passengers
    });
  } catch (error) {
    res.status(500).json({ error: 'Error en la búsqueda' });
  }
});

// GET - Estadísticas
router.get('/stats/summary', (req, res) => {
  try {
    const total = passengersData.passengers.length;
    const survived = passengersData.passengers.filter(p => p.Survived === 1).length;
    const byClass = {
      1: passengersData.passengers.filter(p => p.Pclass === 1).length,
      2: passengersData.passengers.filter(p => p.Pclass === 2).length,
      3: passengersData.passengers.filter(p => p.Pclass === 3).length
    };
    const byGender = {
      male: passengersData.passengers.filter(p => p.Sex === 'male').length,
      female: passengersData.passengers.filter(p => p.Sex === 'female').length
    };

    res.json({
      totalPassengers: total,
      survivalRate: `${((survived / total) * 100).toFixed(2)}%`,
      byClass,
      byGender
    });
  } catch (error) {
    res.status(500).json({ error: 'Error al calcular estadísticas' });
  }
});

module.exports = router;