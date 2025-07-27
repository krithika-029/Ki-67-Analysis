import React, { useState, useEffect } from 'react';

const SimpleConnectionTest = () => {
  const [status, setStatus] = useState('Loading...');
  const [data, setData] = useState(null);

  useEffect(() => {
    const testConnection = async () => {
      try {
        console.log('🚀 Testing connection...');
        
        // Direct fetch calls like the working debug test
        const healthResponse = await fetch('http://localhost:5001/api/health');
        const healthData = await healthResponse.json();
        console.log('✅ Health data:', healthData);
        
        const modelsResponse = await fetch('http://localhost:5001/api/models');
        const modelsData = await modelsResponse.json();
        console.log('✅ Models data:', modelsData);
        
        const ensembleResponse = await fetch('http://localhost:5001/api/ensemble/info');
        const ensembleData = await ensembleResponse.json();
        console.log('✅ Ensemble data:', ensembleData);
        
        setData({
          health: healthData,
          models: modelsData,
          ensemble: ensembleData
        });
        setStatus('Connected');
        
      } catch (error) {
        console.error('❌ Connection failed:', error);
        setStatus(`Error: ${error.message}`);
      }
    };
    
    testConnection();
  }, []);

  return (
    <div style={{ padding: '20px' }}>
      <h1>Simple Connection Test</h1>
      <p>Status: {status}</p>
      
      {data && (
        <div>
          <h2>✅ Connection Successful!</h2>
          <p>Models Loaded: {data.health.models_loaded}/{data.models.total_models}</p>
          <p>Ensemble Accuracy: {data.ensemble.ensemble.high_confidence_accuracy}%</p>
          <p>System Status: {data.health.status}</p>
          
          <h3>Models:</h3>
          <ul>
            {data.models.available_models.map(model => (
              <li key={model.name}>
                {model.name} - {model.accuracy}% accuracy, {model.ensemble_weight} weight
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default SimpleConnectionTest;
