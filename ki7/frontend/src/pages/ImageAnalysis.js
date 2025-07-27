import React, { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  CloudArrowUpIcon,
  PhotoIcon,
  BeakerIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ArrowPathIcon,
  CursorArrowRaysIcon,
  ChartBarIcon,
} from '@heroicons/react/24/outline';
import AnalysisResults from '../components/AnalysisResults';
import CellCountingTool from '../components/CellCountingTool';
import { analyzeImage } from '../services/api';

const ImageAnalysis = () => {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [error, setError] = useState(null);
  const [selectedModels, setSelectedModels] = useState(['enhanced_ensemble']);
  const [activeMode, setActiveMode] = useState('analysis'); // 'analysis' or 'counting'
  const [manualCells, setManualCells] = useState([]);
  const [availableModels, setAvailableModels] = useState([]);
  const [modelsLoading, setModelsLoading] = useState(true);

  const modes = [
    { id: 'analysis', name: 'AI Analysis', icon: BeakerIcon, description: 'Automated Ki-67 detection using AI models' },
    { id: 'counting', name: 'Manual Counting', icon: CursorArrowRaysIcon, description: 'Manual cell marking and counting' },
  ];

  // Fetch available models on component mount
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/models');
        const data = await response.json();
        
        if (data.models) {
          // Format models for frontend use
          const formattedModels = data.models.map(model => ({
            id: model.id,
            name: model.name,
            accuracy: model.accuracy,
            recommended: model.recommended || false,
            type: model.type || 'individual',
            modelsCount: model.models_count
          }));
          
          setAvailableModels(formattedModels);
          
          // Set default selection to recommended ensemble if available
          const recommendedModel = formattedModels.find(m => m.recommended);
          if (recommendedModel) {
            setSelectedModels([recommendedModel.id]);
          }
        }
      } catch (err) {
        console.error('Failed to fetch models:', err);
        // Fallback to default models if API fails
        setAvailableModels([
          { id: 'enhanced_ensemble', name: 'Enhanced Ensemble', accuracy: 94.2, recommended: true, type: 'ensemble' },
          { id: 'efficientnetb2', name: 'EfficientNet-B2', accuracy: 93.2, type: 'advanced' },
          { id: 'regnetyy8gf', name: 'RegNet-Y-8GF', accuracy: 91.7, type: 'advanced' },
        ]);
      } finally {
        setModelsLoading(false);
      }
    };

    fetchModels();
  }, []);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => {
        setUploadedImage({
          file,
          preview: reader.result,
          name: file.name,
          size: file.size
        });
        setAnalysisResults(null);
        setError(null);
      };
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
    },
    multiple: false,
    maxSize: 50 * 1024 * 1024 // 50MB
  });

  const handleAnalyze = async () => {
    if (!uploadedImage) return;

    setIsAnalyzing(true);
    setError(null);

    try {
      const results = await analyzeImage(uploadedImage.file, selectedModels);
      setAnalysisResults(results);
    } catch (err) {
      setError(err.message || 'Analysis failed. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleModelToggle = (modelId) => {
    const selectedModel = availableModels.find(m => m.id === modelId);
    
    if (selectedModel?.type === 'ensemble') {
      // If ensemble is selected, only allow that ensemble
      setSelectedModels([modelId]);
    } else {
      // If individual model is selected, remove any ensembles and toggle the model
      let newSelection = selectedModels.filter(id => {
        const model = availableModels.find(m => m.id === id);
        return model?.type !== 'ensemble';
      });
      
      if (newSelection.includes(modelId)) {
        newSelection = newSelection.filter(id => id !== modelId);
      } else {
        newSelection.push(modelId);
      }
      
      // If no models selected, default to recommended ensemble or first available
      if (newSelection.length === 0) {
        const recommendedModel = availableModels.find(m => m.recommended) || availableModels[0];
        setSelectedModels(recommendedModel ? [recommendedModel.id] : []);
      } else {
        setSelectedModels(newSelection);
      }
    }
  };

  const handleManualCellsUpdate = (cells) => {
    setManualCells(cells);
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-8">
      {/* Main Analysis Container */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
        {/* Header Section */}
        <div className="px-6 sm:px-8 py-8 border-b border-gray-200 bg-gradient-to-r from-blue-50 to-indigo-50">
          <div className="text-center">
            <h1 className="text-3xl sm:text-4xl font-bold text-gray-900 mb-3">Ki-67 Image Analysis</h1>
            <p className="text-lg text-gray-600 leading-relaxed">
              Upload histopathological images for automated Ki-67 protein detection and analysis
            </p>
          </div>
        </div>

        <div className="px-6 sm:px-8 py-8">
          {/* Mode Selection */}
          <div className="mb-10">
            <div className="text-center mb-6">
              <h3 className="text-2xl font-bold text-gray-900 mb-2">Analysis Mode</h3>
              <p className="text-gray-600">Choose your preferred analysis method</p>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {modes.map((mode) => (
                <div
                  key={mode.id}
                  className={`
                    relative p-6 lg:p-8 rounded-2xl border-2 cursor-pointer transition-all duration-300 hover:shadow-lg transform hover:scale-105
                    ${activeMode === mode.id
                      ? 'border-primary-500 bg-primary-50 shadow-lg ring-2 ring-primary-200'
                      : 'border-gray-200 hover:border-gray-300 bg-white'
                    }
                  `}
                  onClick={() => setActiveMode(mode.id)}
                >
                  <div className="flex items-center space-x-4">
                    <div className={`
                      flex-shrink-0 p-3 rounded-xl
                      ${activeMode === mode.id ? 'bg-primary-100' : 'bg-gray-100'}
                    `}>
                      <mode.icon className={`h-8 w-8 ${
                        activeMode === mode.id ? 'text-primary-600' : 'text-gray-400'
                      }`} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h4 className={`text-xl font-bold mb-2 ${
                        activeMode === mode.id ? 'text-primary-900' : 'text-gray-900'
                      }`}>
                        {mode.name}
                      </h4>
                      <p className={`text-sm leading-relaxed ${
                        activeMode === mode.id ? 'text-primary-700' : 'text-gray-600'
                      }`}>
                        {mode.description}
                      </p>
                    </div>
                    {activeMode === mode.id && (
                      <div className="flex-shrink-0">
                        <CheckCircleIcon className="h-8 w-8 text-primary-600" />
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Image Upload Section */}
          <div className="mb-10">
            <div className="text-center mb-6">
              <h3 className="text-2xl font-bold text-gray-900 mb-2">Upload Image</h3>
              <p className="text-gray-600">Select a histopathological image for analysis</p>
            </div>
            <div className="">
              <div
                {...getRootProps()}
                className={`
                  relative border-2 border-dashed rounded-2xl transition-all duration-300 cursor-pointer
                  ${isDragActive 
                    ? 'border-primary-400 bg-primary-50 scale-[1.02] shadow-lg' 
                    : uploadedImage
                      ? 'border-green-400 bg-green-50 hover:bg-green-100'
                      : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
                  }
                `}
              >
                <input {...getInputProps()} />
                
                <div className="p-12">
                  {uploadedImage ? (
                    <div className="space-y-8">
                      <div className="flex justify-center">
                        <CheckCircleIcon className="h-20 w-20 text-green-500" />
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold text-green-700 mb-3">Image uploaded successfully!</p>
                        <p className="text-gray-600 mb-6 text-lg">{uploadedImage.name}</p>
                        <p className="text-gray-500">({formatFileSize(uploadedImage.size)})</p>
                      </div>
                      <div className="flex justify-center">
                        <div className="relative">
                          <img
                            src={uploadedImage.preview}
                            alt="Uploaded"
                            className="max-h-80 rounded-xl shadow-lg border border-gray-200"
                          />
                          <div className="absolute top-3 right-3 bg-green-600 text-white px-4 py-2 rounded-lg text-sm font-semibold">
                            Ready for analysis
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center space-y-8">
                      <div className="flex justify-center">
                        <CloudArrowUpIcon className="h-20 w-20 text-gray-400" />
                      </div>
                      <div className="space-y-4">
                        <p className="text-2xl font-bold text-gray-700">
                          {isDragActive ? 'Drop your image here' : 'Upload medical image'}
                        </p>
                        <p className="text-gray-500 text-lg">
                          Drag and drop or click to select histopathological images
                        </p>
                        <p className="text-gray-400">
                          Supported formats: PNG, JPG, JPEG, TIFF, BMP • Maximum size: 50MB
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Model Selection */}
          {uploadedImage && activeMode === 'analysis' && (
            <div className="mb-10">
              <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-gray-900 mb-2">Select Analysis Models</h3>
                <p className="text-gray-600">Choose one or more AI models for analysis</p>
                {modelsLoading && (
                  <p className="text-sm text-blue-600 mt-2">Loading available models...</p>
                )}
              </div>
              <div className="">
                {modelsLoading ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                    {[1, 2, 3].map((i) => (
                      <div key={i} className="animate-pulse">
                        <div className="bg-gray-200 rounded-2xl h-24"></div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                    {availableModels.map((model) => (
                      <div
                        key={model.id}
                        className={`
                          relative p-6 rounded-2xl border-2 cursor-pointer transition-all duration-300 hover:shadow-lg transform hover:scale-105
                          ${selectedModels.includes(model.id)
                            ? 'border-primary-500 bg-primary-50 shadow-lg'
                            : 'border-gray-200 hover:border-gray-300 bg-white'
                          }
                          ${model.recommended ? 'ring-2 ring-yellow-400 ring-opacity-50' : ''}
                        `}
                        onClick={() => handleModelToggle(model.id)}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center space-x-2 mb-2">
                              <p className="font-semibold text-gray-900 truncate">{model.name}</p>
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
                            <p className="text-sm text-gray-600 mt-1">Accuracy: {model.accuracy}%</p>
                            {model.modelsCount && (
                              <p className="text-xs text-gray-500 mt-1">{model.modelsCount} models combined</p>
                            )}
                            {model.recommended && (
                              <span className="inline-block mt-2 px-2 py-1 text-xs bg-yellow-100 text-yellow-800 rounded-full font-medium">
                                Recommended
                              </span>
                            )}
                          </div>
                          <div className={`
                            flex-shrink-0 w-6 h-6 rounded-full border-2 flex items-center justify-center ml-4
                            ${selectedModels.includes(model.id)
                              ? 'border-primary-500 bg-primary-500'
                              : 'border-gray-300'
                            }
                          `}>
                            {selectedModels.includes(model.id) && (
                              <div className="w-2 h-2 bg-white rounded-full"></div>
                            )}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Analysis Button */}
          {uploadedImage && activeMode === 'analysis' && (
            <div className="text-center mb-10">
              <button
                onClick={handleAnalyze}
                disabled={isAnalyzing || selectedModels.length === 0}
                className={`
                  inline-flex items-center px-12 py-5 text-xl font-bold rounded-2xl transition-all duration-300 shadow-lg
                  ${isAnalyzing || selectedModels.length === 0
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed shadow-none'
                    : 'bg-gradient-to-r from-primary-600 to-medical-600 text-white hover:from-primary-700 hover:to-medical-700 transform hover:scale-105 hover:shadow-xl'
                  }
                `}
              >
                {isAnalyzing ? (
                  <>
                    <ArrowPathIcon className="h-7 w-7 mr-4 animate-spin" />
                    Analyzing Image...
                  </>
                ) : (
                  <>
                    <BeakerIcon className="h-7 w-7 mr-4" />
                    Start Analysis
                  </>
                )}
              </button>
              {selectedModels.length === 0 && (
                <p className="mt-4 text-sm text-red-600 font-semibold">Please select at least one model</p>
              )}
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="p-6 bg-red-50 border border-red-200 rounded-xl mb-10">
              <div className="flex items-center">
                <ExclamationTriangleIcon className="h-6 w-6 text-red-500 mr-3 flex-shrink-0" />
                <p className="text-red-700 font-semibold">{error}</p>
              </div>
            </div>
          )}

          {/* Analysis Progress */}
          {isAnalyzing && (
            <div className="p-10 bg-blue-50 rounded-2xl border border-blue-200 mb-10">
              <div className="text-center">
                <div className="inline-flex items-center justify-center w-20 h-20 bg-blue-500 rounded-full mb-8">
                  <BeakerIcon className="h-10 w-10 text-white animate-pulse" />
                </div>
                <h3 className="text-2xl font-bold text-blue-900 mb-4">Analysis in Progress</h3>
                <p className="text-blue-700 mb-8 text-lg leading-relaxed">
                  Our AI models are analyzing your image for Ki-67 protein expression...
                </p>
                <div className="w-full bg-blue-200 rounded-full h-4 mb-6">
                  <div className="bg-blue-500 h-4 rounded-full animate-pulse transition-all duration-1000" style={{ width: '45%' }}></div>
                </div>
                <p className="text-sm text-blue-600 font-medium">Expected completion: ~15 seconds</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Manual Counting Tool */}
      {activeMode === 'counting' && uploadedImage && (
        <CellCountingTool 
          originalImage={uploadedImage} 
          onCellsUpdate={handleManualCellsUpdate}
        />
      )}

      {/* AI Analysis Results */}
      {activeMode === 'analysis' && analysisResults && (
        <div className="mt-10">
          <AnalysisResults results={analysisResults} originalImage={uploadedImage} />
        </div>
      )}

      {/* Manual Counting Results */}
      {activeMode === 'counting' && manualCells.length > 0 && (
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden">
          <div className="px-6 sm:px-8 py-6 border-b border-gray-200 bg-gradient-to-r from-indigo-50 to-purple-50">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-3 sm:space-y-0">
              <h3 className="text-xl lg:text-2xl font-bold text-gray-900 flex items-center">
                <ChartBarIcon className="h-6 lg:h-7 w-6 lg:w-7 mr-3 text-indigo-600" />
                Manual Counting Results
              </h3>
              <div className="text-sm text-gray-500 bg-white px-3 py-1.5 rounded-full border">
                {manualCells.length} cells manually marked
              </div>
            </div>
          </div>

          <div className="px-6 sm:px-8 py-6 lg:py-8">
            {/* Statistics Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 lg:gap-6 mb-8">
              <div className="bg-gradient-to-br from-red-50 to-red-100 rounded-2xl p-5 lg:p-6 text-center border border-red-200">
                <div className="text-3xl lg:text-4xl font-bold text-red-600 mb-2">
                  {manualCells.filter(c => c.type === 'positive').length}
                </div>
                <p className="text-red-700 font-semibold text-base lg:text-lg">Ki-67 Positive</p>
                <p className="text-red-600 text-xs lg:text-sm mt-1">Proliferating cells</p>
              </div>
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-2xl p-5 lg:p-6 text-center border border-blue-200">
                <div className="text-3xl lg:text-4xl font-bold text-blue-600 mb-2">
                  {manualCells.filter(c => c.type === 'negative').length}
                </div>
                <p className="text-blue-700 font-semibold text-base lg:text-lg">Ki-67 Negative</p>
                <p className="text-blue-600 text-xs lg:text-sm mt-1">Non-proliferating cells</p>
              </div>
              <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-2xl p-5 lg:p-6 text-center border border-purple-200">
                <div className="text-3xl lg:text-4xl font-bold text-purple-600 mb-2">
                  {manualCells.filter(c => c.type === 'mitotic').length}
                </div>
                <p className="text-purple-700 font-semibold text-base lg:text-lg">Mitotic Figures</p>
                <p className="text-purple-600 text-xs lg:text-sm mt-1">Dividing cells</p>
              </div>
              <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-2xl p-5 lg:p-6 text-center border border-green-200">
                <div className="text-3xl lg:text-4xl font-bold text-green-600 mb-2">
                  {manualCells.length > 0 ? (
                    ((manualCells.filter(c => c.type === 'positive').length / 
                      Math.max(manualCells.filter(c => c.type !== 'unclear').length, 1)) * 100).toFixed(1)
                  ) : '0.0'}%
                </div>
                <p className="text-green-700 font-semibold text-base lg:text-lg">Ki-67 Index</p>
                <p className="text-green-600 text-xs lg:text-sm mt-1">Proliferation rate</p>
              </div>
            </div>

            {/* Manual vs AI Comparison */}
            {analysisResults && (
              <div className="bg-gray-50 rounded-2xl p-6 lg:p-8 border border-gray-200">
                <h4 className="text-xl font-semibold text-gray-900 mb-6 text-center">Manual vs AI Comparison</h4>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8">
                  <div className="text-center">
                    <div className="bg-white rounded-xl p-5 lg:p-6 shadow-sm border border-gray-200">
                      <p className="text-sm font-medium text-gray-600 mb-2">Manual Count</p>
                      <p className="text-2xl lg:text-3xl font-bold text-indigo-600 mb-1">
                        {((manualCells.filter(c => c.type === 'positive').length / 
                          Math.max(manualCells.filter(c => c.type !== 'unclear').length, 1)) * 100).toFixed(1)}%
                      </p>
                      <p className="text-xs text-gray-500">Ki-67 Index</p>
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="bg-white rounded-xl p-5 lg:p-6 shadow-sm border border-gray-200">
                      <p className="text-sm font-medium text-gray-600 mb-2">AI Analysis</p>
                      <p className="text-2xl lg:text-3xl font-bold text-green-600 mb-1">{analysisResults.ki67Index}%</p>
                      <p className="text-xs text-gray-500">Ki-67 Index</p>
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="bg-white rounded-xl p-5 lg:p-6 shadow-sm border border-gray-200">
                      <p className="text-sm font-medium text-gray-600 mb-2">Difference</p>
                      <p className="text-2xl lg:text-3xl font-bold text-orange-600 mb-1">
                        ±{Math.abs(
                          ((manualCells.filter(c => c.type === 'positive').length / 
                            Math.max(manualCells.filter(c => c.type !== 'unclear').length, 1)) * 100) - 
                          analysisResults.ki67Index
                        ).toFixed(1)}%
                      </p>
                      <p className="text-xs text-gray-500">Absolute difference</p>
                    </div>
                  </div>
                </div>
                
                {/* Agreement Assessment */}
                <div className="mt-6 text-center">
                  {Math.abs(
                    ((manualCells.filter(c => c.type === 'positive').length / 
                      Math.max(manualCells.filter(c => c.type !== 'unclear').length, 1)) * 100) - 
                    analysisResults.ki67Index
                  ) <= 5 ? (
                    <div className="inline-flex items-center px-4 py-2 bg-green-100 text-green-800 rounded-full text-sm font-medium">
                      <CheckCircleIcon className="h-4 w-4 mr-2" />
                      Excellent Agreement (≤5% difference)
                    </div>
                  ) : Math.abs(
                    ((manualCells.filter(c => c.type === 'positive').length / 
                      Math.max(manualCells.filter(c => c.type !== 'unclear').length, 1)) * 100) - 
                    analysisResults.ki67Index
                  ) <= 10 ? (
                    <div className="inline-flex items-center px-4 py-2 bg-yellow-100 text-yellow-800 rounded-full text-sm font-medium">
                      <ExclamationTriangleIcon className="h-4 w-4 mr-2" />
                      Good Agreement (≤10% difference)
                    </div>
                  ) : (
                    <div className="inline-flex items-center px-4 py-2 bg-red-100 text-red-800 rounded-full text-sm font-medium">
                      <ExclamationTriangleIcon className="h-4 w-4 mr-2" />
                      Poor Agreement ({'>'}10% difference)
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageAnalysis;
