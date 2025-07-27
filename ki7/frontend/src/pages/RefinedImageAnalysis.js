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
  SparklesIcon,
  AdjustmentsHorizontalIcon,
} from '@heroicons/react/24/outline';
import AnalysisResults from '../components/AnalysisResults';
import { analyzeImage, getModels, checkHealth } from '../services/refined_api';
import { simpleFetchHealth, simpleFetchModels, simpleFetchAnalyze } from '../services/simple_api';

const RefinedImageAnalysis = () => {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [error, setError] = useState(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.7);
  const [systemStatus, setSystemStatus] = useState(null);

  // Cleanup blob URLs when component unmounts
  useEffect(() => {
    return () => {
      if (uploadedImage?.preview && uploadedImage.preview.startsWith('blob:')) {
        URL.revokeObjectURL(uploadedImage.preview);
      }
      if (uploadedImage?.blobUrl) {
        URL.revokeObjectURL(uploadedImage.blobUrl);
      }
    };
  }, [uploadedImage?.preview, uploadedImage?.blobUrl]);
  const [ensembleInfo, setEnsembleInfo] = useState(null);

  // Fetch system status and ensemble info
  useEffect(() => {
    const fetchSystemInfo = async () => {
      try {
        console.log('🔍 Image Analysis: Fetching system info...');
        const [healthData, modelsData] = await Promise.all([
          simpleFetchHealth(),
          simpleFetchModels()
        ]);
        
        console.log('✅ Image Analysis: Health data:', healthData);
        console.log('✅ Image Analysis: Models data:', modelsData);
        
        setSystemStatus(healthData);
        setEnsembleInfo(modelsData.ensemble_info);
        
        // Only clear error if we successfully got data
        if (healthData && modelsData) {
          setError(null);
        }
      } catch (error) {
        console.error('❌ Image Analysis: Failed to fetch system info:', error);
        // Only set error if it's a real connectivity issue
        if (error.message.includes('fetch') || error.message.includes('network') || error.message.includes('Failed to fetch')) {
          setError('Backend connection failed - running in demo mode');
        }
      }
    };

    fetchSystemInfo();
  }, []);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      // Clean up previous blob URL if it exists
      if (uploadedImage?.preview && uploadedImage.preview.startsWith('blob:')) {
        URL.revokeObjectURL(uploadedImage.preview);
      }
      
      // Create both blob URL and data URL for redundancy
      const blobUrl = URL.createObjectURL(file);
      
      // Also create a data URL as backup
      const reader = new FileReader();
      reader.onload = function(e) {
        const dataUrl = e.target.result;
        
        setUploadedImage(prev => {
          // Clean up previous blob URL if switching
          if (prev?.preview && prev.preview.startsWith('blob:')) {
            URL.revokeObjectURL(prev.preview);
          }
          
          return {
            file: file,
            preview: dataUrl, // Use data URL instead of blob URL
            blobUrl: blobUrl, // Keep blob URL as backup
            name: file.name,
            size: file.size,
            type: file.type
          };
        });
        
        console.log('New image uploaded with data URL:', {
          name: file.name,
          hasDataUrl: !!dataUrl,
          hasBlobUrl: !!blobUrl
        });
      };
      
      reader.readAsDataURL(file);
      
      // Set initial state with blob URL until data URL is ready
      const newImageData = {
        file: file,
        preview: blobUrl,
        name: file.name,
        size: file.size,
        type: file.type
      };
      
      console.log('New image uploaded (initial):', newImageData);
      setUploadedImage(newImageData);
      setAnalysisResults(null);
      setError(null);
    }
  }, [uploadedImage?.preview]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.tiff', '.bmp']
    },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024 // 50MB
  });

  const handleAnalyze = async () => {
    console.log('🔘 Analyze button clicked!');
    console.log('📁 Uploaded image:', uploadedImage);
    
    if (!uploadedImage) {
      console.log('❌ No uploaded image');
      return;
    }

    console.log('🔍 Starting image analysis...');
    setIsAnalyzing(true);
    setError(null);

    try {
      const results = await simpleFetchAnalyze(uploadedImage.file, confidenceThreshold);
      console.log('✅ Analysis completed:', results);
      setAnalysisResults(results);
    } catch (error) {
      console.error('❌ Analysis failed:', error);
      setError(error.message);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    // Clean up blob URLs
    if (uploadedImage?.preview && uploadedImage.preview.startsWith('blob:')) {
      URL.revokeObjectURL(uploadedImage.preview);
    }
    if (uploadedImage?.blobUrl) {
      URL.revokeObjectURL(uploadedImage.blobUrl);
    }
    
    setUploadedImage(null);
    setAnalysisResults(null);
    setError(null);
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-10">
      {/* Header */}
      <div className="text-center lg:text-left">
        <div className="max-w-3xl mx-auto lg:mx-0">
          <h1 className="text-4xl lg:text-5xl font-bold text-gray-900 mb-4">
            Ki-67 Image Analysis
          </h1>
          <p className="text-xl text-gray-600 leading-relaxed">
            Advanced AI-powered analysis for Ki-67 proliferation marker detection using our refined ensemble model
          </p>
        </div>
      </div>

      {/* System Status */}
      {systemStatus && (
        <div className={`flex items-center justify-center gap-2 px-4 py-3 rounded-full text-sm font-medium mx-auto w-fit ${
          systemStatus.status === 'operational' 
            ? 'bg-green-100 text-green-800' 
            : 'bg-orange-100 text-orange-800'
        }`}>
          <div className={`w-2 h-2 rounded-full ${
            systemStatus.status === 'operational' ? 'bg-green-500' : 'bg-orange-500'
          }`}></div>
          {systemStatus.ensemble_ready ? 'Ensemble Ready' : 'Limited Models'}
        </div>
      )}

      {/* Ensemble Info Panel */}
      {ensembleInfo && (
        <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl p-6 border border-blue-100">
          <div className="flex items-center gap-3 mb-4">
            <SparklesIcon className="h-6 w-6 text-purple-600" />
            <h2 className="text-xl font-bold text-gray-900">{ensembleInfo.name}</h2>
            <span className="px-3 py-1 bg-purple-100 text-purple-800 text-sm font-medium rounded-full">
              {ensembleInfo.high_confidence_accuracy}% Accuracy
            </span>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <span className="font-medium text-gray-600">Coverage:</span>
              <span className="ml-2 text-gray-900">{ensembleInfo.coverage}% of samples</span>
            </div>
            <div>
              <span className="font-medium text-gray-600">Models:</span>
              <span className="ml-2 text-gray-900">{ensembleInfo.loaded_models} active</span>
            </div>
            <div>
              <span className="font-medium text-gray-600">Threshold:</span>
              <span className="ml-2 text-gray-900">{ensembleInfo.optimal_threshold}</span>
            </div>
          </div>
        </div>
      )}

      {/* Confidence Threshold Control */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-100 p-6">
        <div className="flex items-center gap-3 mb-4">
          <AdjustmentsHorizontalIcon className="h-5 w-5 text-gray-600" />
          <h3 className="text-lg font-semibold text-gray-900">Analysis Settings</h3>
        </div>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Confidence Threshold: {confidenceThreshold}
            </label>
            <input
              type="range"
              min="0.3"
              max="0.9"
              step="0.1"
              value={confidenceThreshold}
              onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>0.3 (More Coverage)</span>
              <span>0.9 (Higher Precision)</span>
            </div>
          </div>
          
          <div className="text-sm text-gray-600">
            <p>
              <strong>Higher thresholds</strong> provide more accurate predictions on fewer samples.
              <strong className="ml-1">Lower thresholds</strong> analyze more samples with slightly lower accuracy.
            </p>
          </div>
        </div>
      </div>

      {/* Main Analysis Panel */}
      <div className="grid grid-cols-1 xl:grid-cols-5 gap-8 lg:gap-12">
        {/* Upload Section */}
        <div className="xl:col-span-2 space-y-8">
          <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 hover:shadow-md transition-shadow duration-200">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center">
                <CloudArrowUpIcon className="h-5 w-5 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900">Image Upload</h3>
            </div>
            
            {!uploadedImage ? (
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-all duration-200 ${
                  isDragActive
                    ? 'border-blue-400 bg-blue-50'
                    : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
                }`}
              >
                <input {...getInputProps()} />
                <CloudArrowUpIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-lg font-medium text-gray-900 mb-2">
                  {isDragActive ? 'Drop the image here' : 'Upload histopathology image'}
                </p>
                <p className="text-gray-600 mb-4">
                  Drag and drop an image, or click to select
                </p>
                <p className="text-sm text-gray-500">
                  Supports: JPEG, PNG, TIFF, BMP (max 50MB)
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                {/* Image Preview */}
                <div className="relative bg-gray-50 rounded-xl p-4">
                  <img
                    src={uploadedImage.preview}
                    alt="Uploaded image"
                    className="w-full h-72 object-contain rounded-lg"
                  />
                </div>
                
                {/* File Information */}
                <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-6 border border-gray-200">
                  <h4 className="font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <PhotoIcon className="h-5 w-5 text-gray-600" />
                    File Information
                  </h4>
                  <div className="grid grid-cols-1 gap-3 text-sm">
                    <div className="flex justify-between items-center py-2 border-b border-gray-200 last:border-b-0">
                      <span className="font-medium text-gray-600">Name:</span>
                      <span className="text-gray-900 font-mono text-xs bg-gray-200 px-2 py-1 rounded">
                        {uploadedImage.name}
                      </span>
                    </div>
                    <div className="flex justify-between items-center py-2 border-b border-gray-200 last:border-b-0">
                      <span className="font-medium text-gray-600">Size:</span>
                      <span className="text-gray-900">{formatFileSize(uploadedImage.size)}</span>
                    </div>
                    <div className="flex justify-between items-center py-2">
                      <span className="font-medium text-gray-600">Type:</span>
                      <span className="text-gray-900">{uploadedImage.type}</span>
                    </div>
                  </div>
                </div>

                {/* Action Buttons */}
                <div className="flex gap-4">
                  <button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className="flex-1 flex items-center justify-center gap-3 bg-gradient-to-r from-blue-600 to-blue-700 text-white px-6 py-4 rounded-xl font-semibold hover:from-blue-700 hover:to-blue-800 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transform hover:scale-105"
                  >
                    {isAnalyzing ? (
                      <>
                        <ArrowPathIcon className="h-5 w-5 animate-spin" />
                        Analyzing Image...
                      </>
                    ) : (
                      <>
                        <BeakerIcon className="h-5 w-5" />
                        Analyze Image
                      </>
                    )}
                  </button>
                  
                  <button
                    onClick={handleReset}
                    className="px-6 py-4 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-xl transition-all duration-200 font-medium border border-gray-300 hover:border-gray-400"
                  >
                    Reset
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Results Section */}
        <div className="xl:col-span-3 space-y-8">
          {error && !systemStatus && (
            <div className="bg-gradient-to-br from-red-50 to-red-100 border border-red-200 rounded-xl p-6 shadow-sm">
              <div className="flex items-start gap-4">
                <div className="w-10 h-10 bg-red-500 rounded-full flex items-center justify-center flex-shrink-0">
                  <ExclamationTriangleIcon className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h3 className="font-semibold text-red-900 mb-2">Analysis Error</h3>
                  <p className="text-red-700 leading-relaxed">{error}</p>
                </div>
              </div>
            </div>
          )}

          {isAnalyzing && (
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 border border-blue-200 rounded-xl p-8 shadow-sm">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 bg-blue-500 rounded-full flex items-center justify-center">
                  <ArrowPathIcon className="h-6 w-6 text-white animate-spin" />
                </div>
                <div>
                  <h3 className="font-semibold text-blue-900 mb-2">Processing Image</h3>
                  <p className="text-blue-700 leading-relaxed">
                    Running refined ensemble analysis with 3 AI models...
                  </p>
                  <div className="mt-3 bg-blue-200 rounded-full h-2 overflow-hidden">
                    <div className="bg-blue-600 h-full w-3/4 rounded-full animate-pulse"></div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {analysisResults && (
            <div className="bg-white rounded-xl shadow-lg border border-gray-100 overflow-hidden">
              <div className="bg-gradient-to-r from-green-500 to-emerald-600 px-6 py-6 text-white">
                <div className="flex items-center gap-4">
                  <div className="w-12 h-12 bg-white bg-opacity-20 rounded-full flex items-center justify-center">
                    <CheckCircleIcon className="h-6 w-6 text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold">Analysis Complete</h3>
                    <p className="text-green-100 mt-1">
                      Ki-67 protein expression analysis for {uploadedImage?.name || 'uploaded image'}
                    </p>
                  </div>
                </div>
              </div>
              
              <AnalysisResults results={analysisResults} originalImage={uploadedImage} />
            </div>
          )}

          {!analysisResults && !isAnalyzing && !error && (
            <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-12 text-center border border-gray-200">
              <div className="w-16 h-16 bg-gray-300 rounded-full flex items-center justify-center mx-auto mb-6">
                <PhotoIcon className="h-8 w-8 text-gray-500" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-3">Ready for Analysis</h3>
              <p className="text-gray-600 leading-relaxed max-w-md mx-auto">
                Upload a medical image to start AI-powered Ki-67 proliferation analysis with our refined ensemble model
              </p>
              <div className="mt-6 flex items-center justify-center gap-2 text-sm text-gray-500">
                <SparklesIcon className="h-4 w-4" />
                <span>3 AI models • 97.4% accuracy</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RefinedImageAnalysis;
