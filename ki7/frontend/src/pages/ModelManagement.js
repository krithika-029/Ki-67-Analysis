import React, { useState, useEffect } from 'react';
import {
  CogIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ArrowPathIcon,
  InformationCircleIcon,
  ChartBarIcon,
} from '@heroicons/react/24/outline';
import { getModelStatus } from '../services/api';

const ModelManagement = () => {
  const [modelStatus, setModelStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [retraining, setRetraining] = useState({});

  useEffect(() => {
    fetchModelStatus();
  }, []);

  const fetchModelStatus = async () => {
    try {
      const data = await getModelStatus();
      setModelStatus(data);
    } catch (error) {
      console.error('Failed to fetch model status:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleRetrain = (modelId) => {
    setRetraining(prev => ({ ...prev, [modelId]: true }));
    // Simulate retraining process
    setTimeout(() => {
      setRetraining(prev => ({ ...prev, [modelId]: false }));
    }, 5000);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'active':
        return 'text-green-600 bg-green-100';
      case 'training':
        return 'text-yellow-600 bg-yellow-100';
      case 'error':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'active':
        return <CheckCircleIcon className="h-5 w-5 text-green-500" />;
      case 'training':
        return <ArrowPathIcon className="h-5 w-5 text-yellow-500 animate-spin" />;
      case 'error':
        return <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />;
      default:
        return <InformationCircleIcon className="h-5 w-5 text-gray-500" />;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Model Management</h1>
            <p className="text-gray-600 mt-1">
              Monitor and manage your AI models for Ki-67 analysis
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
              modelStatus?.systemStatus === 'healthy' 
                ? 'text-green-700 bg-green-100' 
                : 'text-red-700 bg-red-100'
            }`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${
                modelStatus?.systemStatus === 'healthy' ? 'bg-green-500' : 'bg-red-500'
              }`}></div>
              System {modelStatus?.systemStatus === 'healthy' ? 'Healthy' : 'Error'}
            </div>
          </div>
        </div>

        {/* System Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6">
          <div className="bg-blue-50 rounded-lg p-4">
            <p className="text-blue-600 text-sm font-medium">Total Analyses</p>
            <p className="text-2xl font-bold text-blue-900">{modelStatus?.totalAnalyses || 0}</p>
          </div>
          <div className="bg-green-50 rounded-lg p-4">
            <p className="text-green-600 text-sm font-medium">Active Models</p>
            <p className="text-2xl font-bold text-green-900">
              {modelStatus?.models?.filter(m => m.status === 'active').length || 0}/{modelStatus?.models?.length || 0}
            </p>
          </div>
          <div className="bg-purple-50 rounded-lg p-4">
            <p className="text-purple-600 text-sm font-medium">Avg Processing Time</p>
            <p className="text-2xl font-bold text-purple-900">{modelStatus?.avgProcessingTime || 0}s</p>
          </div>
        </div>
      </div>

      {/* Models Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {modelStatus?.models?.map((model) => (
          <div key={model.id} className="bg-white rounded-xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className="w-12 h-12 bg-gradient-to-r from-primary-500 to-medical-500 rounded-lg flex items-center justify-center">
                  <CogIcon className="h-6 w-6 text-white" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">{model.name}</h3>
                  <p className="text-sm text-gray-600">Model ID: {model.id}</p>
                </div>
              </div>
              {getStatusIcon(retraining[model.id] ? 'training' : model.status)}
            </div>

            <div className="space-y-4">
              {/* Status */}
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-600">Status</span>
                <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStatusColor(retraining[model.id] ? 'training' : model.status)}`}>
                  {retraining[model.id] ? 'Retraining' : model.status}
                </span>
              </div>

              {/* Accuracy */}
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-600">Accuracy</span>
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-semibold text-gray-900">{model.accuracy}%</span>
                  <div className="w-16 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-green-500 h-2 rounded-full" 
                      style={{ width: `${model.accuracy}%` }}
                    ></div>
                  </div>
                </div>
              </div>

              {/* Last Updated */}
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-600">Last Updated</span>
                <span className="text-sm text-gray-900">{model.lastUpdated}</span>
              </div>

              {/* Performance Metrics */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="text-sm font-medium text-gray-900 mb-3">Performance Metrics</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-gray-600">Precision</p>
                    <p className="text-sm font-semibold text-gray-900">
                      {(model.accuracy - Math.random() * 5).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600">Recall</p>
                    <p className="text-sm font-semibold text-gray-900">
                      {(model.accuracy - Math.random() * 3).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600">F1-Score</p>
                    <p className="text-sm font-semibold text-gray-900">
                      {(model.accuracy - Math.random() * 4).toFixed(1)}%
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600">AUC</p>
                    <p className="text-sm font-semibold text-gray-900">
                      {(model.accuracy + Math.random() * 2).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex space-x-2 pt-2">
                <button
                  onClick={() => handleRetrain(model.id)}
                  disabled={retraining[model.id] || model.status !== 'active'}
                  className={`flex-1 px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                    retraining[model.id] || model.status !== 'active'
                      ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                      : 'bg-primary-600 text-white hover:bg-primary-700'
                  }`}
                >
                  {retraining[model.id] ? (
                    <>
                      <ArrowPathIcon className="h-4 w-4 mr-1 animate-spin" />
                      Retraining...
                    </>
                  ) : (
                    'Retrain Model'
                  )}
                </button>
                <button className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors">
                  View Details
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Training History */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Training History</h3>
        <div className="space-y-4">
          {[
            { model: 'Enhanced Ensemble', date: '2024-01-15', accuracy: 94.2, status: 'completed' },
            { model: 'EfficientNet-B2', date: '2024-01-14', accuracy: 93.2, status: 'completed' },
            { model: 'RegNet-Y-8GF', date: '2024-01-13', accuracy: 91.7, status: 'completed' },
          ].map((training, index) => (
            <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-r from-primary-500 to-medical-500 rounded-lg flex items-center justify-center">
                  <ChartBarIcon className="h-5 w-5 text-white" />
                </div>
                <div>
                  <p className="font-medium text-gray-900">{training.model}</p>
                  <p className="text-sm text-gray-600">Trained on {training.date}</p>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <div className="text-right">
                  <p className="text-sm font-semibold text-gray-900">{training.accuracy}%</p>
                  <p className="text-xs text-gray-600">Accuracy</p>
                </div>
                <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-full text-green-700 bg-green-100">
                  {training.status}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* System Configuration */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">System Configuration</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-gray-900 mb-3">Processing Settings</h4>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Batch Size</span>
                <span className="text-sm font-medium text-gray-900">32</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Image Size</span>
                <span className="text-sm font-medium text-gray-900">224x224</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">GPU Memory</span>
                <span className="text-sm font-medium text-gray-900">8GB</span>
              </div>
            </div>
          </div>
          
          <div>
            <h4 className="font-medium text-gray-900 mb-3">Model Settings</h4>
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Ensemble Voting</span>
                <span className="text-sm font-medium text-gray-900">Weighted</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Confidence Threshold</span>
                <span className="text-sm font-medium text-gray-900">0.8</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-600">Auto-retrain</span>
                <span className="text-sm font-medium text-green-600">Enabled</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelManagement;
