import React, { useState } from 'react';
import {
  ChartBarIcon,
  DocumentArrowDownIcon,
  EyeIcon,
  ShareIcon,
  ClockIcon,
  CheckCircleIcon,
  CursorArrowRaysIcon,
  BeakerIcon,
  ChartPieIcon,
  CogIcon,
} from '@heroicons/react/24/outline';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import ImageAnnotationViewer from './ImageAnnotationViewer';

const AnalysisResults = ({ results, originalImage }) => {
  const [activeTab, setActiveTab] = useState('overview');

  const tabs = [
    { id: 'overview', name: 'Overview', icon: ChartBarIcon },
    { id: 'annotation', name: 'Cell Annotation', icon: CursorArrowRaysIcon },
    { id: 'details', name: 'Detailed Analysis', icon: EyeIcon },
    { id: 'statistics', name: 'Advanced Statistics', icon: ChartPieIcon },
    { id: 'comparison', name: 'Model Comparison', icon: BeakerIcon },
  ];

  const handleDownloadReport = () => {
    // Implement PDF report generation
    console.log('Downloading report...');
  };

  const handleShare = () => {
    // Implement sharing functionality
    console.log('Sharing results...');
  };

  if (!results) return null;

  return (
    <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-green-500 to-emerald-600 px-6 sm:px-8 py-8 text-white">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6">
          <div className="flex-1">
            <h2 className="text-3xl font-bold mb-3">Analysis Complete</h2>
            <p className="text-green-100 text-base leading-relaxed">
              Ki-67 protein expression analysis for <span className="font-semibold">{originalImage?.name || 'uploaded image'}</span>
            </p>
          </div>
          <div className="flex items-center justify-center lg:justify-end space-x-8">
            <div className="text-center">
              <CheckCircleIcon className="h-12 w-12 mx-auto mb-2" />
              <p className="text-sm font-semibold">Completed</p>
            </div>
            <div className="text-center">
              <ClockIcon className="h-12 w-12 mx-auto mb-2" />
              <p className="text-sm font-semibold">{results.processingTime}s</p>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="border-b border-gray-200 bg-gray-50">
        <nav className="flex overflow-x-auto px-6 sm:px-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`
                flex items-center px-6 py-4 border-b-2 font-semibold text-sm transition-all duration-300 whitespace-nowrap min-w-max
                ${activeTab === tab.id
                  ? 'border-primary-500 text-primary-600 bg-white shadow-sm'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 hover:bg-white/50'
                }
              `}
            >
              <tab.icon className="h-5 w-5 mr-3" />
              {tab.name}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab Content */}
      <div className="px-6 sm:px-8 py-8">
        {activeTab === 'overview' && (
          <div className="space-y-10">
            {/* Key Metrics */}
            <div>
              <h3 className="text-2xl font-bold text-gray-900 mb-6">Key Performance Metrics</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-2xl p-8 border border-blue-200 transform hover:scale-105 transition-transform duration-200">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-semibold text-blue-900">Ki-67 Index</h4>
                    <div className="w-10 h-10 bg-blue-500 rounded-full flex items-center justify-center">
                      <ChartBarIcon className="h-5 w-5 text-white" />
                    </div>
                  </div>
                  <div className="text-4xl font-bold text-blue-700 mb-2">
                    {results.ki67Index}%
                  </div>
                  <p className="text-blue-600 text-sm font-medium leading-relaxed">
                    Percentage of Ki-67 positive cells detected in the tissue sample
                  </p>
                </div>
                
                <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-2xl p-8 border border-green-200 transform hover:scale-105 transition-transform duration-200">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-semibold text-green-900">Confidence</h4>
                    <div className="w-10 h-10 bg-green-500 rounded-full flex items-center justify-center">
                      <CheckCircleIcon className="h-5 w-5 text-white" />
                    </div>
                  </div>
                  <div className="text-4xl font-bold text-green-700 mb-2">
                    {Math.round(results.confidence * 100)}%
                  </div>
                  <p className="text-green-600 text-sm font-medium leading-relaxed">
                    Model prediction confidence level for this analysis
                  </p>
                </div>
                
                <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-2xl p-8 border border-purple-200 transform hover:scale-105 transition-transform duration-200">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-semibold text-purple-900">Cell Count</h4>
                    <div className="w-10 h-10 bg-purple-500 rounded-full flex items-center justify-center">
                      <EyeIcon className="h-5 w-5 text-white" />
                    </div>
                  </div>
                  <div className="text-4xl font-bold text-purple-700 mb-2">
                    {results.totalCells}
                  </div>
                  <p className="text-purple-600 text-sm font-medium leading-relaxed">
                    Total number of cells analyzed in the image
                  </p>
                </div>
              </div>
            </div>

            {/* Visual Results */}
            <div>
              <h3 className="text-2xl font-bold text-gray-900 mb-6">Image Analysis Results</h3>
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
                <div className="bg-gray-50 rounded-2xl p-6 border border-gray-200">
                  <div className="flex items-center justify-between mb-6">
                    <h4 className="text-xl font-semibold text-gray-900">Original Image</h4>
                    <div className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                      Source
                    </div>
                  </div>
                  <div className="relative rounded-xl overflow-hidden shadow-lg bg-white">
                    {originalImage?.preview ? (
                      <img
                        src={originalImage.preview}
                        alt="Original uploaded image"
                        className="w-full h-auto"
                        onLoad={() => console.log('✅ Original image loaded successfully')}
                        onError={(e) => {
                          console.error('❌ Original image load error:', e);
                          console.error('Image src:', originalImage.preview);
                          // Try to show fallback
                          e.target.style.display = 'none';
                          e.target.nextSibling.style.display = 'flex';
                        }}
                      />
                    ) : null}
                    {/* Fallback message */}
                    <div 
                      className="w-full h-48 bg-gray-200 flex items-center justify-center"
                      style={{ display: originalImage?.preview ? 'none' : 'flex' }}
                    >
                      <div className="text-center">
                        <span className="text-gray-500 block">Original image not available</span>
                        {originalImage && (
                          <span className="text-xs text-gray-400 mt-1 block">
                            File: {originalImage.name || 'Unknown'}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
                
                <div className="bg-gray-50 rounded-2xl p-6 border border-gray-200">
                  <div className="flex items-center justify-between mb-6">
                    <h4 className="text-xl font-semibold text-gray-900">Analysis Overlay</h4>
                    <div className="px-3 py-1 bg-green-100 text-green-800 rounded-full text-sm font-medium">
                      Processed
                    </div>
                  </div>
                  <div className="relative rounded-xl overflow-hidden shadow-lg bg-white">
                    {originalImage?.preview ? (
                      <img
                        src={results.annotatedImage || originalImage.preview}
                        alt="Analyzed image with Ki-67 annotations"
                        className="w-full h-auto"
                        onLoad={() => console.log('✅ Analysis image loaded successfully')}
                        onError={(e) => {
                          console.error('❌ Analysis image load error:', e);
                          console.error('Analysis image src:', results.annotatedImage || originalImage.preview);
                          // Try to show fallback
                          e.target.style.display = 'none';
                          e.target.nextSibling.style.display = 'flex';
                        }}
                      />
                    ) : null}
                    {/* Fallback message */}
                    <div 
                      className="w-full h-48 bg-gray-200 flex items-center justify-center"
                      style={{ display: originalImage?.preview ? 'none' : 'flex' }}
                    >
                      <div className="text-center">
                        <span className="text-gray-500 block">Analysis image not available</span>
                        {originalImage && (
                          <span className="text-xs text-gray-400 mt-1 block">
                            File: {originalImage.name || 'Unknown'}
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="absolute top-4 right-4 bg-black bg-opacity-80 text-white px-4 py-2 rounded-lg text-sm font-semibold backdrop-blur-sm">
                      Ki-67+ cells highlighted
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Clinical Interpretation */}
            <div>
              <h3 className="text-2xl font-bold text-gray-900 mb-6">Clinical Interpretation</h3>
              <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-2xl p-8 border border-gray-200">
                <div className="flex items-center mb-6">
                  <BeakerIcon className="h-8 w-8 mr-3 text-medical-600" />
                  <h4 className="text-xl font-semibold text-gray-900">Analysis Summary</h4>
                </div>
                <div className="space-y-6">
                  <div className="flex items-start space-x-4 p-4 bg-white rounded-xl border border-gray-200">
                    <div className="w-4 h-4 bg-blue-500 rounded-full mt-1 flex-shrink-0"></div>
                    <div className="flex-1">
                      <p className="text-gray-700 leading-relaxed">
                        <span className="font-semibold text-gray-900">Ki-67 Index:</span> {results.ki67Index}% indicates{' '}
                        <span className={`font-bold ${
                          results.ki67Index > 20 ? 'text-red-600' : 
                          results.ki67Index > 10 ? 'text-orange-600' : 'text-green-600'
                        }`}>
                          {results.ki67Index > 20 ? 'high' : results.ki67Index > 10 ? 'moderate' : 'low'}
                        </span>{' '}
                        proliferative activity in the tissue sample.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-4 p-4 bg-white rounded-xl border border-gray-200">
                    <div className="w-4 h-4 bg-green-500 rounded-full mt-1 flex-shrink-0"></div>
                    <div className="flex-1">
                      <p className="text-gray-700 leading-relaxed">
                        <span className="font-semibold text-gray-900">Confidence Level:</span> The analysis shows {results.confidence}% confidence, indicating{' '}
                        <span className="font-semibold text-green-600">reliable results</span> from the AI model.
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start space-x-4 p-4 bg-white rounded-xl border border-gray-200">
                    <div className="w-4 h-4 bg-purple-500 rounded-full mt-1 flex-shrink-0"></div>
                    <div className="flex-1">
                      <p className="text-gray-700 leading-relaxed">
                        <span className="font-semibold text-gray-900">Cell Population:</span> Analysis of {results.totalCells} total cells identified{' '}
                        <span className="font-bold text-red-600">{results.positiveCells}</span> Ki-67 positive cells and{' '}
                        <span className="font-bold text-blue-600">{results.negativeCells}</span> negative cells.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'details' && (
          <div className="space-y-10">
            <div>
              <h3 className="text-2xl font-bold text-gray-900 mb-6">Detailed Analysis Results</h3>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="space-y-6">
                  <div className="flex items-center space-x-3">
                    <EyeIcon className="h-8 w-8 text-blue-600" />
                    <h4 className="text-xl font-semibold text-gray-900">Detection Details</h4>
                  </div>
                  
                  <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-2xl p-6 border border-blue-200">
                    <div className="space-y-5">
                      <div className="flex justify-between items-center py-3 border-b border-blue-200">
                        <span className="font-semibold text-blue-900">Total Cells</span>
                        <span className="text-blue-900 font-bold text-lg">{results.totalCells}</span>
                      </div>
                      <div className="flex justify-between items-center py-3 border-b border-blue-200">
                        <span className="font-semibold text-blue-900">Ki-67 Positive</span>
                        <span className="text-green-600 font-bold text-lg">{results.positiveCells}</span>
                      </div>
                      <div className="flex justify-between items-center py-3 border-b border-blue-200">
                        <span className="font-semibold text-blue-900">Ki-67 Negative</span>
                        <span className="text-gray-600 font-bold text-lg">{results.negativeCells}</span>
                      </div>
                      <div className="flex justify-between items-center py-3">
                        <span className="font-semibold text-blue-900">Ki-67 Index</span>
                        <span className="text-blue-700 font-bold text-2xl">{results.ki67Index}%</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="space-y-6">
                  <div className="flex items-center space-x-3">
                    <CogIcon className="h-8 w-8 text-purple-600" />
                    <h4 className="text-xl font-semibold text-gray-900">Technical Details</h4>
                  </div>
                  
                  <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-2xl p-6 border border-purple-200">
                    <div className="space-y-5">
                      <div className="flex justify-between items-center py-3 border-b border-purple-200">
                        <span className="font-semibold text-purple-900">Model Used</span>
                        <span className="text-purple-900 font-medium">{results.modelUsed}</span>
                      </div>
                      <div className="flex justify-between items-center py-3 border-b border-purple-200">
                        <span className="font-semibold text-purple-900">Processing Time</span>
                        <span className="text-purple-900 font-medium">{results.processingTime}s</span>
                      </div>
                      <div className="flex justify-between items-center py-3 border-b border-purple-200">
                        <span className="font-semibold text-purple-900">Image Resolution</span>
                        <span className="text-purple-900 font-medium">{results.imageResolution}</span>
                      </div>
                      <div className="flex justify-between items-center py-3">
                        <span className="font-semibold text-purple-900">Analysis Date</span>
                        <span className="text-purple-900 font-medium">{new Date().toLocaleDateString()}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Regional Analysis */}
            <div>
              <div className="flex items-center space-x-3 mb-6">
                <ChartBarIcon className="h-8 w-8 text-green-600" />
                <h4 className="text-xl font-semibold text-gray-900">Regional Analysis</h4>
              </div>
              <div className="bg-white rounded-2xl border border-gray-200 overflow-hidden shadow-sm">
                <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gradient-to-r from-green-50 to-green-100">
                      <tr>
                        <th className="px-8 py-5 text-left text-sm font-bold text-green-900 uppercase tracking-wider">
                          Region
                        </th>
                        <th className="px-8 py-5 text-left text-sm font-bold text-green-900 uppercase tracking-wider">
                          Total Cells
                        </th>
                        <th className="px-8 py-5 text-left text-sm font-bold text-green-900 uppercase tracking-wider">
                          Ki-67 Positive
                        </th>
                        <th className="px-8 py-5 text-left text-sm font-bold text-green-900 uppercase tracking-wider">
                          Ki-67 Index
                        </th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {results.regionalAnalysis?.map((region, index) => (
                        <tr key={index} className="hover:bg-green-50 transition-colors duration-200">
                          <td className="px-8 py-5 whitespace-nowrap text-sm font-semibold text-gray-900">
                            {region.name}
                          </td>
                          <td className="px-8 py-5 whitespace-nowrap text-sm text-gray-600 font-medium">
                            {region.totalCells}
                          </td>
                          <td className="px-8 py-5 whitespace-nowrap text-sm font-bold text-green-600">
                            {region.positiveCells}
                          </td>
                          <td className="px-8 py-5 whitespace-nowrap text-sm font-bold text-blue-600">
                            {region.ki67Index}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'comparison' && (
          <div className="space-y-8">
            <h3 className="text-xl font-semibold text-gray-900 flex items-center">
              <BeakerIcon className="h-6 w-6 mr-2 text-purple-600" />
              Model Performance Comparison
            </h3>
            
            <div className="space-y-6">
              {results.modelComparison?.map((model, index) => (
                <div key={index} className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-6 border border-gray-200">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-6 gap-4">
                    <h4 className="text-lg font-semibold text-gray-900">{model.name}</h4>
                    <div className="flex flex-wrap items-center gap-4">
                      <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-blue-100 text-blue-800">
                        Accuracy: {model.accuracy}%
                      </span>
                      <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
                        Agreement: {model.agreementScore}%
                      </span>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
                    <div className="text-center bg-white rounded-lg p-4 border border-gray-200">
                      <p className="text-3xl font-bold text-blue-600 mb-1">{model.ki67Index}%</p>
                      <p className="text-sm text-gray-600 font-medium">Ki-67 Index</p>
                    </div>
                    <div className="text-center bg-white rounded-lg p-4 border border-gray-200">
                      <p className="text-3xl font-bold text-green-600 mb-1">{model.confidence}%</p>
                      <p className="text-sm text-gray-600 font-medium">Confidence</p>
                    </div>
                    <div className="text-center bg-white rounded-lg p-4 border border-gray-200">
                      <p className="text-3xl font-bold text-purple-600 mb-1">{model.processingTime}s</p>
                      <p className="text-sm text-gray-600 font-medium">Time</p>
                    </div>
                    <div className="text-center bg-white rounded-lg p-4 border border-gray-200">
                      <p className="text-3xl font-bold text-orange-600 mb-1">{model.cellsDetected}</p>
                      <p className="text-sm text-gray-600 font-medium">Cells</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'annotation' && (
          <div className="space-y-6">
            <h3 className="text-xl font-semibold text-gray-900 mb-6 flex items-center">
              <CursorArrowRaysIcon className="h-6 w-6 mr-2 text-indigo-600" />
              Interactive Cell Annotation
            </h3>
            <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-6 border border-gray-200">
              <ImageAnnotationViewer originalImage={originalImage} analysisResults={results} />
            </div>
          </div>
        )}

        {activeTab === 'statistics' && (
          <div className="space-y-8">
            <h3 className="text-xl font-semibold text-gray-900 mb-6 flex items-center">
              <ChartPieIcon className="h-6 w-6 mr-2 text-indigo-600" />
              Advanced Statistical Analysis
            </h3>
            
            {/* Cell Density Statistics */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
                <h4 className="text-lg font-semibold text-gray-900 mb-6 flex items-center">
                  <ChartBarIcon className="h-5 w-5 mr-2 text-blue-600" />
                  Cell Density Analysis
                </h4>
                <div className="space-y-4">
                  <div className="flex justify-between items-center py-3 border-b border-gray-100">
                    <span className="text-gray-600 font-medium">Total Density (cells/mm²)</span>
                    <span className="font-bold text-gray-900">{results.statistics?.cell_density_per_mm2}</span>
                  </div>
                  <div className="flex justify-between items-center py-3 border-b border-gray-100">
                    <span className="text-gray-600 font-medium">Positive Cell Density</span>
                    <span className="font-bold text-red-600">{results.statistics?.positive_cell_density}</span>
                  </div>
                  <div className="flex justify-between items-center py-3 border-b border-gray-100">
                    <span className="text-gray-600 font-medium">Negative Cell Density</span>
                    <span className="font-bold text-blue-600">{results.statistics?.negative_cell_density}</span>
                  </div>
                  <div className="flex justify-between items-center py-3">
                    <span className="text-gray-600 font-medium">Average Cell Size</span>
                    <span className="font-bold text-gray-900">{results.statistics?.average_cell_size} px²</span>
                  </div>
                </div>
              </div>

              <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
                <h4 className="text-lg font-semibold text-gray-900 mb-6 flex items-center">
                  <ChartPieIcon className="h-5 w-5 mr-2 text-purple-600" />
                  Staining Intensity Distribution
                </h4>
                <ResponsiveContainer width="100%" height={250}>
                  <PieChart>
                    <Pie
                      data={[
                        { name: 'Strong', value: results.statistics?.staining_intensity_distribution?.strong || 0, fill: '#dc2626' },
                        { name: 'Moderate', value: results.statistics?.staining_intensity_distribution?.moderate || 0, fill: '#ea580c' },
                        { name: 'Weak', value: results.statistics?.staining_intensity_distribution?.weak || 0, fill: '#facc15' }
                      ]}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                    />
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* ROI Analysis */}
            <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
              <h4 className="text-lg font-semibold text-gray-900 mb-6 flex items-center">
                <EyeIcon className="h-5 w-5 mr-2 text-green-600" />
                Region of Interest Analysis
              </h4>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={results.roiRegions || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f3f4f6" />
                  <XAxis 
                    dataKey="id" 
                    tickFormatter={(value) => `ROI-${value}`}
                    tick={{ fontSize: 12 }}
                  />
                  <YAxis tick={{ fontSize: 12 }} />
                  <Tooltip 
                    formatter={(value, name) => [value, name === 'proliferation_index' ? 'Ki-67 Index (%)' : name]}
                    contentStyle={{
                      backgroundColor: '#f9fafb',
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px'
                    }}
                  />
                  <Legend />
                  <Bar dataKey="proliferation_index" fill="#8884d8" name="Ki-67 Index (%)" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="cell_count" fill="#82ca9d" name="Total Cells" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Hot Spots */}
            <div className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
              <h4 className="text-lg font-semibold text-gray-900 mb-6 flex items-center">
                <CursorArrowRaysIcon className="h-5 w-5 mr-2 text-red-600" />
                Proliferation Hot Spots
              </h4>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gradient-to-r from-gray-50 to-gray-100">
                    <tr>
                      <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700 uppercase tracking-wider">
                        Hot Spot
                      </th>
                      <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700 uppercase tracking-wider">
                        Location (x, y)
                      </th>
                      <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700 uppercase tracking-wider">
                        Ki-67 %
                      </th>
                      <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700 uppercase tracking-wider">
                        Cell Density
                      </th>
                      <th className="px-6 py-4 text-left text-sm font-semibold text-gray-700 uppercase tracking-wider">
                        Significance
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {results.hotSpots?.map((spot, index) => (
                      <tr key={index} className="hover:bg-gray-50 transition-colors">
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                          Hot Spot {spot.id}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                          ({spot.x}, {spot.y})
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-red-600">
                          {spot.ki67_percentage}%
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                          {spot.cell_density} cells/mm²
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`inline-flex px-3 py-1 text-xs font-semibold rounded-full ${
                            spot.significance === 'high' 
                              ? 'bg-red-100 text-red-800 border border-red-200' 
                              : 'bg-yellow-100 text-yellow-800 border border-yellow-200'
                          }`}>
                            {spot.significance}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Quality Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-xl p-6 border border-blue-200">
                <h5 className="font-semibold text-blue-900 mb-3 text-sm uppercase tracking-wide">Image Quality</h5>
                <p className="text-3xl font-bold text-blue-700">{results.qualityMetrics?.imageQuality}</p>
              </div>
              <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-xl p-6 border border-green-200">
                <h5 className="font-semibold text-green-900 mb-3 text-sm uppercase tracking-wide">Staining Quality</h5>
                <p className="text-3xl font-bold text-green-700">{results.qualityMetrics?.stainingQuality}</p>
              </div>
              <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-xl p-6 border border-purple-200">
                <h5 className="font-semibold text-purple-900 mb-3 text-sm uppercase tracking-wide">Focus Quality</h5>
                <p className="text-3xl font-bold text-purple-700">{results.qualityMetrics?.focusQuality}</p>
              </div>
              <div className="bg-gradient-to-br from-orange-50 to-orange-100 rounded-xl p-6 border border-orange-200">
                <h5 className="font-semibold text-orange-900 mb-3 text-sm uppercase tracking-wide">Artifact Level</h5>
                <p className="text-3xl font-bold text-orange-700">{results.qualityMetrics?.artifactLevel}</p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Action Buttons */}
      <div className="bg-gradient-to-r from-gray-50 to-gray-100 px-6 sm:px-8 py-8 border-t border-gray-200">
        <div className="flex flex-col lg:flex-row justify-between items-center gap-6">
          <div className="flex flex-wrap gap-4">
            <button
              onClick={handleDownloadReport}
              className="inline-flex items-center px-8 py-4 border border-gray-300 shadow-sm text-sm font-semibold rounded-xl text-gray-700 bg-white hover:bg-gray-50 hover:border-gray-400 hover:shadow-md transform hover:scale-105 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <DocumentArrowDownIcon className="h-5 w-5 mr-3" />
              Download Report
            </button>
            <button
              onClick={handleShare}
              className="inline-flex items-center px-8 py-4 border border-gray-300 shadow-sm text-sm font-semibold rounded-xl text-gray-700 bg-white hover:bg-gray-50 hover:border-gray-400 hover:shadow-md transform hover:scale-105 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <ShareIcon className="h-5 w-5 mr-3" />
              Share Results
            </button>
          </div>
          
          <div className="flex items-center text-sm text-gray-500 bg-white px-4 py-2 rounded-lg border border-gray-200">
            <ClockIcon className="h-4 w-4 mr-2" />
            Analysis completed at {new Date().toLocaleTimeString()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisResults;
