import React, { useState, useEffect } from 'react';
import { checkHealth, getModels, getEnsembleInfo } from '../services/refined_api';

const LiveDataTest = () => {
  const [data, setData] = useState({
    health: null,
    models: null,
    ensemble: null,
    error: null,
    loading: true
  });

  useEffect(() => {
    const fetchData = async () => {
      try {
        console.log('🔄 Starting API tests...');
        
        const health = await checkHealth();
        console.log('Health:', health);
        
        const models = await getModels();
        console.log('Models:', models);
        
        const ensemble = await getEnsembleInfo();
        console.log('Ensemble:', ensemble);
        
        setData({
          health,
          models,
          ensemble,
          error: null,
          loading: false
        });
        
        console.log('✅ All API calls successful');
      } catch (error) {
        console.error('❌ API Error:', error);
        setData(prev => ({
          ...prev,
          error: error.message,
          loading: false
        }));
      }
    };
    
    fetchData();
  }, []);

  if (data.loading) {
    return <div className="p-4">Loading API data...</div>;
  }

  if (data.error) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded">
        <h3 className="font-bold text-red-800">API Error</h3>
        <p className="text-red-600">{data.error}</p>
      </div>
    );
  }

  return (
    <div className="p-4 space-y-4">
      <h2 className="text-xl font-bold">Live API Data Test</h2>
      
      <div className="bg-green-50 border border-green-200 rounded p-4">
        <h3 className="font-bold text-green-800">Health Data</h3>
        <div className="text-sm">
          <p>Status: {data.health?.status}</p>
          <p>Models Loaded: {data.health?.models_loaded}</p>
          <p>Total Models: {data.health?.total_models}</p>
          <p>Ensemble Ready: {data.health?.ensemble_ready ? 'Yes' : 'No'}</p>
        </div>
      </div>
      
      <div className="bg-blue-50 border border-blue-200 rounded p-4">
        <h3 className="font-bold text-blue-800">Models Data</h3>
        <div className="text-sm">
          <p>Total Models: {data.models?.total_models}</p>
          <p>Available: {data.models?.available_models?.length}</p>
          <p>Ensemble Accuracy: {data.models?.ensemble_info?.high_confidence_accuracy}%</p>
        </div>
      </div>
      
      <div className="bg-purple-50 border border-purple-200 rounded p-4">
        <h3 className="font-bold text-purple-800">Ensemble Data</h3>
        <div className="text-sm">
          <p>Name: {data.ensemble?.ensemble?.name}</p>
          <p>Accuracy: {data.ensemble?.ensemble?.high_confidence_accuracy}%</p>
          <p>Coverage: {data.ensemble?.ensemble?.coverage}%</p>
          <p>Models: {data.ensemble?.ensemble?.loaded_models}</p>
        </div>
      </div>
      
      <div className="bg-gray-50 border border-gray-200 rounded p-4">
        <h3 className="font-bold text-gray-800">Calculated Dashboard Stats</h3>
        <div className="text-sm">
          <p>Models Ready: {data.health?.models_loaded}/{data.models?.total_models}</p>
          <p>Avg Accuracy: {data.ensemble?.ensemble?.high_confidence_accuracy || 97.4}%</p>
          <p>System Status: {data.health?.status === 'operational' ? 'Operational' : 'Limited'}</p>
        </div>
      </div>
    </div>
  );
};

export default LiveDataTest;
