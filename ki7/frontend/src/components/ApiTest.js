import React, { useState, useEffect } from 'react';
import { checkHealth, getModels, getEnsembleInfo } from '../services/refined_api';

const ApiTest = () => {
  const [results, setResults] = useState({
    health: null,
    models: null,
    ensemble: null,
    errors: []
  });

  useEffect(() => {
    const testApis = async () => {
      const errors = [];
      let healthData = null;
      let modelsData = null;
      let ensembleData = null;

      try {
        console.log('🔍 Testing health API...');
        healthData = await checkHealth();
        console.log('✅ Health result:', healthData);
      } catch (error) {
        console.error('❌ Health error:', error);
        errors.push(`Health API: ${error.message}`);
      }

      try {
        console.log('🔍 Testing models API...');
        modelsData = await getModels();
        console.log('✅ Models result:', modelsData);
      } catch (error) {
        console.error('❌ Models error:', error);
        errors.push(`Models API: ${error.message}`);
      }

      try {
        console.log('🔍 Testing ensemble API...');
        ensembleData = await getEnsembleInfo();
        console.log('✅ Ensemble result:', ensembleData);
      } catch (error) {
        console.error('❌ Ensemble error:', error);
        errors.push(`Ensemble API: ${error.message}`);
      }

      setResults({
        health: healthData,
        models: modelsData,
        ensemble: ensembleData,
        errors
      });
    };

    testApis();
  }, []);

  return (
    <div style={{ padding: '20px', fontFamily: 'monospace' }}>
      <h1>🧪 API Connection Test</h1>
      
      <div style={{ marginBottom: '20px' }}>
        <h2>Health API:</h2>
        {results.health ? (
          <pre style={{ background: '#e8f5e8', padding: '10px' }}>
            {JSON.stringify(results.health, null, 2)}
          </pre>
        ) : (
          <div>Loading...</div>
        )}
      </div>

      <div style={{ marginBottom: '20px' }}>
        <h2>Models API:</h2>
        {results.models ? (
          <div>
            <p>✅ Total Models: {results.models.total_models}</p>
            <p>✅ Loaded Models: {results.models.ensemble_info?.loaded_models}</p>
            <pre style={{ background: '#e8f5e8', padding: '10px', fontSize: '12px' }}>
              {JSON.stringify(results.models, null, 2)}
            </pre>
          </div>
        ) : (
          <div>Loading...</div>
        )}
      </div>

      <div style={{ marginBottom: '20px' }}>
        <h2>Ensemble API:</h2>
        {results.ensemble ? (
          <div>
            <p>✅ Ensemble Models: {results.ensemble.ensemble?.loaded_models}</p>
            <pre style={{ background: '#e8f5e8', padding: '10px', fontSize: '12px' }}>
              {JSON.stringify(results.ensemble, null, 2)}
            </pre>
          </div>
        ) : (
          <div>Loading...</div>
        )}
      </div>

      {results.errors.length > 0 && (
        <div style={{ marginBottom: '20px' }}>
          <h2>❌ Errors:</h2>
          {results.errors.map((error, index) => (
            <div key={index} style={{ background: '#ffe8e8', padding: '10px', margin: '5px 0' }}>
              {error}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default ApiTest;
