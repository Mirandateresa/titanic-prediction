const express = require('express');
const cors = require('cors');
const passengerRoutes = require('./routes/passengers');

const app = express();
const PORT = process.env.PORT || 3000;

// Middlewares
app.use(cors());
app.use(express.json());

// Rutas
app.use('/api/passengers', passengerRoutes);

// Ruta de prueba
app.get('/', (req, res) => {
  res.json({ 
    message: 'API del Titanic funcionando',
    endpoints: {
      'GET /api/passengers': 'Obtener todos los pasajeros',
      'GET /api/passengers/:id': 'Obtener pasajero por ID',
      'GET /api/passengers/class/:class': 'Filtrar por clase',
      'GET /api/passengers/survived/:status': 'Filtrar por supervivencia',
      'GET /api/passengers/search/:name': 'Buscar por nombre'
    }
  });
});

// Iniciar servidor
app.listen(PORT, () => {
  console.log(` API del Titanic escuchando en puerto ${PORT}`);
});