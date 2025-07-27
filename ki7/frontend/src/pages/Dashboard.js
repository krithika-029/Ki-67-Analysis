import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  ChartBarIcon,
  CameraIcon,
  BeakerIcon,
  ClockIcon,
  CheckCircleIcon,
  ArrowTrendingUpIcon,
} from '@heroicons/react/24/outline';

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalAnalyses: 0,
    modelsReady: 0,
    avgAccuracy: 0,
    recentAnalyses: [],
    modelStats: null
  });

  useEffect(() => {
    // Fetch model information from API
    const fetchModelStats = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/models');
        const data = await response.json();
        
        if (data.models) {
          const ensembleModels = data.models.filter(m => m.type === 'ensemble');
          const individualModels = data.models.filter(m => m.type !== 'ensemble');
          const bestModel = data.models.reduce((best, current) => 
            current.accuracy > (best?.accuracy || 0) ? current : best, null
          );
          
          setStats(prevStats => ({
            ...prevStats,
            modelsReady: data.totalModels || data.models.length,
            avgAccuracy: bestModel?.accuracy || 94.2,
            modelStats: {
              totalModels: data.totalModels,
              individualModels: data.individualModels || individualModels.length,
              ensembleModels: data.ensembleModels || ensembleModels.length,
              models: data.models,
              bestModel: bestModel
            }
          }));
        }
      } catch (error) {
        console.error('Failed to fetch model stats:', error);
        // Fallback stats
        setStats(prevStats => ({
          ...prevStats,
          modelsReady: 6,
          avgAccuracy: 94.2
        }));
      }
    };

    // Simulate loading other stats
    setTimeout(() => {
      setStats(prevStats => ({
        ...prevStats,
        totalAnalyses: 1247,
        recentAnalyses: [
          { id: 1, name: 'Sample_001.png', accuracy: 96.4, date: '2 minutes ago', status: 'completed' },
          { id: 2, name: 'Sample_002.png', accuracy: 92.1, date: '1 hour ago', status: 'completed' },
          { id: 3, name: 'Sample_003.png', accuracy: 95.8, date: '3 hours ago', status: 'completed' },
        ]
      }));
    }, 1000);

    fetchModelStats();
  }, []);

  const quickStats = [
    {
      title: 'Total Analyses',
      value: stats.totalAnalyses,
      icon: ChartBarIcon,
      color: 'from-blue-500 to-blue-600',
      change: '+12%',
      changeType: 'increase'
    },
    {
      title: 'Models Ready',
      value: `${stats.modelsReady}/${stats.modelStats?.totalModels || stats.modelsReady}`,
      icon: BeakerIcon,
      color: 'from-green-500 to-green-600',
      change: 'All Active',
      changeType: 'neutral'
    },
    {
      title: 'Average Accuracy',
      value: `${stats.avgAccuracy}%`,
      icon: ArrowTrendingUpIcon,
      color: 'from-purple-500 to-purple-600',
      change: '+2.3%',
      changeType: 'increase'
    },
    {
      title: 'Processing Time',
      value: '~15s',
      icon: ClockIcon,
      color: 'from-orange-500 to-orange-600',
      change: '-30%',
      changeType: 'decrease'
    }
  ];

  return (
    <div className="h-full space-y-8 full-width-container">
      {/* Welcome Section */}
      <div className="bg-gradient-to-r from-medical-500 to-primary-600 rounded-2xl p-8 lg:p-12 text-white w-full">
        <div className="flex items-center justify-between w-full">
          <div className="flex-1">
            <h1 className="text-4xl lg:text-5xl font-bold mb-4">Welcome to Ki-67 Analyzer Pro</h1>
            <p className="text-blue-100 text-xl lg:text-2xl leading-relaxed">
              Advanced AI-powered histopathological image analysis for precise Ki-67 detection
            </p>
            <div className="mt-8 flex flex-wrap gap-4">
              <Link
                to="/analyze"
                className="bg-white text-primary-600 px-8 py-4 rounded-xl font-semibold hover:bg-gray-50 transition-colors inline-flex items-center text-lg"
              >
                <CameraIcon className="h-6 w-6 mr-3" />
                Start Analysis
              </Link>
              <Link
                to="/results"
                className="border-2 border-white text-white px-8 py-4 rounded-xl font-semibold hover:bg-white hover:text-primary-600 transition-colors text-lg"
              >
                View Results
              </Link>
            </div>
          </div>
          <div className="hidden xl:block ml-12">
            <div className="w-40 h-40 bg-white bg-opacity-20 rounded-full flex items-center justify-center">
              <BeakerIcon className="h-20 w-20 text-white" />
            </div>
          </div>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="full-width-grid">
        {quickStats.map((stat, index) => (
          <div key={index} className="bg-white rounded-2xl shadow-lg p-8 hover:shadow-xl transition-all duration-300 hover:scale-105 w-full">
            <div className="flex items-center justify-between w-full">
              <div className="flex-1 min-w-0 pr-4">
                <p className="text-gray-600 text-base font-medium">{stat.title}</p>
                <p className="text-3xl lg:text-4xl font-bold text-gray-900 mt-2 truncate">{stat.value}</p>
                <div className="flex items-center mt-3">
                  <span className={`text-base font-medium ${
                    stat.changeType === 'increase' ? 'text-green-600' :
                    stat.changeType === 'decrease' ? 'text-red-600' : 'text-gray-600'
                  }`}>
                    {stat.change}
                  </span>
                </div>
              </div>
              <div className={`w-16 h-16 rounded-xl bg-gradient-to-r ${stat.color} flex items-center justify-center flex-shrink-0`}>
                <stat.icon className="h-8 w-8 text-white" />
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Recent Activity */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8 w-full">
        <div className="bg-white rounded-2xl shadow-lg p-8 w-full">
          <h3 className="text-2xl font-semibold text-gray-900 mb-6">Recent Analyses</h3>
          <div className="space-y-4">
            {stats.recentAnalyses.map((analysis) => (
              <div key={analysis.id} className="flex items-center justify-between p-6 bg-gray-50 rounded-xl hover:bg-gray-100 transition-colors w-full">
                <div className="flex items-center space-x-4 flex-1 min-w-0">
                  <div className="w-12 h-12 bg-gradient-to-r from-primary-500 to-medical-500 rounded-xl flex items-center justify-center flex-shrink-0">
                    <CameraIcon className="h-6 w-6 text-white" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="font-semibold text-gray-900 text-lg truncate">{analysis.name}</p>
                    <p className="text-base text-gray-600">{analysis.date}</p>
                  </div>
                </div>
                <div className="flex items-center space-x-3 flex-shrink-0">
                  <span className="text-lg font-semibold text-gray-900">{analysis.accuracy}%</span>
                  <CheckCircleIcon className="h-6 w-6 text-green-500" />
                </div>
              </div>
            ))}
          </div>
          <Link
            to="/results"
            className="block w-full text-center mt-6 text-primary-600 font-semibold hover:text-primary-700 text-lg py-2"
          >
            View All Results →
          </Link>
        </div>

        {/* Model Status */}
        <div className="bg-white rounded-2xl shadow-lg p-8 w-full">
          <h3 className="text-2xl font-semibold text-gray-900 mb-6">Model Status</h3>
          <div className="space-y-4">
            {stats.modelStats?.models ? (
              stats.modelStats.models
                .sort((a, b) => b.accuracy - a.accuracy)
                .map((model, index) => (
                  <div key={index} className="flex items-center justify-between p-4 bg-gray-50 rounded-xl hover:bg-gray-100 transition-colors w-full">
                    <div className="flex items-center space-x-4 flex-1 min-w-0">
                      <div className="w-3 h-3 bg-green-500 rounded-full flex-shrink-0"></div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2">
                          <span className="font-semibold text-gray-900 text-lg truncate">{model.name}</span>
                          {model.recommended && (
                            <span className="inline-block px-2 py-1 text-xs bg-yellow-100 text-yellow-800 rounded-full font-medium">
                              Recommended
                            </span>
                          )}
                          {model.type === 'ensemble' && (
                            <span className="inline-block px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded-full font-medium">
                              Ensemble
                            </span>
                          )}
                          {model.type === 'advanced' && (
                            <span className="inline-block px-2 py-1 text-xs bg-green-100 text-green-800 rounded-full font-medium">
                              Advanced
                            </span>
                          )}
                        </div>
                        {model.models_count && (
                          <span className="text-xs text-gray-500">{model.models_count} models combined</span>
                        )}
                      </div>
                    </div>
                    <span className="text-base text-gray-600 font-medium flex-shrink-0">{model.accuracy}% acc</span>
                  </div>
                ))
            ) : (
              // Loading skeleton
              [1, 2, 3, 4, 5, 6].map((i) => (
                <div key={i} className="animate-pulse flex items-center justify-between p-4 bg-gray-50 rounded-xl">
                  <div className="flex items-center space-x-4 flex-1">
                    <div className="w-3 h-3 bg-gray-300 rounded-full"></div>
                    <div className="h-4 bg-gray-300 rounded w-32"></div>
                  </div>
                  <div className="h-4 bg-gray-300 rounded w-16"></div>
                </div>
              ))
            )}
          </div>
          <Link
            to="/models"
            className="block w-full text-center mt-6 text-primary-600 font-semibold hover:text-primary-700 text-lg py-2"
          >
            Manage Models →
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
