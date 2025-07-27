import React, { useState, useCallback } from 'react';
import {
  PlusIcon,
  MinusIcon,
  TrashIcon,
  EyeIcon,
  EyeSlashIcon,
  ArrowUturnLeftIcon,
  DocumentDuplicateIcon,
  BeakerIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline';

const CellCountingTool = ({ originalImage, onCellsUpdate }) => {
  const [manualCells, setManualCells] = useState([]);
  const [countingMode, setCountingMode] = useState('positive'); // 'positive', 'negative', 'mitotic'
  const [showAICells, setShowAICells] = useState(true);
  const [showManualCells, setShowManualCells] = useState(true);
  const [isCountingActive, setIsCountingActive] = useState(false);

  const cellTypes = [
    { id: 'positive', name: 'Ki-67 Positive', color: '#dc2626', count: 0 },
    { id: 'negative', name: 'Ki-67 Negative', color: '#2563eb', count: 0 },
    { id: 'mitotic', name: 'Mitotic Figures', color: '#7c3aed', count: 0 },
    { id: 'unclear', name: 'Unclear/Artifact', color: '#6b7280', count: 0 }
  ];

  const getCellCounts = () => {
    const counts = {};
    cellTypes.forEach(type => {
      counts[type.id] = manualCells.filter(cell => cell.type === type.id).length;
    });
    return counts;
  };

  const handleImageClick = useCallback((event) => {
    if (!isCountingActive) return;

    const rect = event.target.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Convert to image coordinates (assuming 1024x1024 base image)
    const imageX = (x / rect.width) * 1024;
    const imageY = (y / rect.height) * 1024;

    const newCell = {
      id: Date.now(),
      x: imageX,
      y: imageY,
      type: countingMode,
      timestamp: new Date().toISOString(),
      confidence: 1.0, // Manual marking = 100% confidence
      manual: true
    };

    setManualCells(prev => [...prev, newCell]);
    
    // Update parent component
    const allCells = [...manualCells, newCell];
    onCellsUpdate && onCellsUpdate(allCells);
  }, [isCountingActive, countingMode, manualCells, onCellsUpdate]);

  const removeLastCell = () => {
    if (manualCells.length > 0) {
      const newCells = manualCells.slice(0, -1);
      setManualCells(newCells);
      onCellsUpdate && onCellsUpdate(newCells);
    }
  };

  const clearAllCells = () => {
    setManualCells([]);
    onCellsUpdate && onCellsUpdate([]);
  };

  const exportCellData = () => {
    const cellData = {
      manual_cells: manualCells,
      counts: getCellCounts(),
      image_name: originalImage?.name || 'unknown',
      timestamp: new Date().toISOString(),
      total_manual_cells: manualCells.length
    };

    const dataStr = JSON.stringify(cellData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `manual_cell_counts_${originalImage?.name || 'image'}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const cellCounts = getCellCounts();
  const totalManualCells = manualCells.length;

  return (
    <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 p-4 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold">Manual Cell Counting</h3>
            <p className="text-indigo-100 text-sm">
              Click to mark cells • {totalManualCells} cells marked
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setIsCountingActive(!isCountingActive)}
              className={`p-2 rounded-lg transition-colors ${
                isCountingActive 
                  ? 'bg-green-600 hover:bg-green-700' 
                  : 'bg-indigo-700 hover:bg-indigo-800'
              }`}
              title={isCountingActive ? 'Stop Counting' : 'Start Counting'}
            >
              {isCountingActive ? (
                <BeakerIcon className="h-5 w-5" />
              ) : (
                <PlusIcon className="h-5 w-5" />
              )}
            </button>
            <button
              onClick={exportCellData}
              className="p-2 bg-indigo-700 hover:bg-indigo-800 rounded-lg transition-colors"
              title="Export Cell Data"
            >
              <DocumentDuplicateIcon className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>

      <div className="flex">
        {/* Controls Panel */}
        <div className="w-80 bg-gray-50 p-4 border-r border-gray-200">
          {/* Cell Type Selection */}
          <div className="mb-6">
            <h4 className="font-semibold text-gray-900 mb-3">Cell Type to Mark</h4>
            <div className="space-y-2">
              {cellTypes.map(type => (
                <label key={type.id} className="flex items-center">
                  <input
                    type="radio"
                    name="cellType"
                    value={type.id}
                    checked={countingMode === type.id}
                    onChange={(e) => setCountingMode(e.target.value)}
                    className="sr-only"
                  />
                  <div className={`w-full p-3 rounded-lg border-2 transition-all cursor-pointer ${
                    countingMode === type.id
                      ? 'border-indigo-500 bg-indigo-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        <div 
                          className="w-4 h-4 rounded-full mr-3"
                          style={{ backgroundColor: type.color }}
                        ></div>
                        <span className="font-medium text-sm">{type.name}</span>
                      </div>
                      <span className="text-lg font-bold" style={{ color: type.color }}>
                        {cellCounts[type.id]}
                      </span>
                    </div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Display Controls */}
          <div className="mb-6">
            <h4 className="font-semibold text-gray-900 mb-3">Display Options</h4>
            <div className="space-y-2">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={showAICells}
                  onChange={(e) => setShowAICells(e.target.checked)}
                  className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                />
                <div className="ml-3 flex items-center">
                  {showAICells ? <EyeIcon className="h-4 w-4 mr-1" /> : <EyeSlashIcon className="h-4 w-4 mr-1" />}
                  <span className="text-sm font-medium">AI Detected Cells</span>
                </div>
              </label>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={showManualCells}
                  onChange={(e) => setShowManualCells(e.target.checked)}
                  className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-500"
                />
                <div className="ml-3 flex items-center">
                  {showManualCells ? <EyeIcon className="h-4 w-4 mr-1" /> : <EyeSlashIcon className="h-4 w-4 mr-1" />}
                  <span className="text-sm font-medium">Manual Marks</span>
                </div>
              </label>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="space-y-3">
            <button
              onClick={removeLastCell}
              disabled={manualCells.length === 0}
              className="w-full flex items-center justify-center px-4 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              <ArrowUturnLeftIcon className="h-4 w-4 mr-2" />
              Undo Last Mark
            </button>
            <button
              onClick={clearAllCells}
              disabled={manualCells.length === 0}
              className="w-full flex items-center justify-center px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors"
            >
              <TrashIcon className="h-4 w-4 mr-2" />
              Clear All Marks
            </button>
          </div>

          {/* Statistics */}
          <div className="mt-6 p-4 bg-white rounded-lg border border-gray-200">
            <h5 className="font-semibold text-gray-900 mb-3 flex items-center">
              <ChartBarIcon className="h-4 w-4 mr-2" />
              Manual Count Summary
            </h5>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Total Marked:</span>
                <span className="font-semibold">{totalManualCells}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Positive:</span>
                <span className="font-semibold text-red-600">{cellCounts.positive}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Negative:</span>
                <span className="font-semibold text-blue-600">{cellCounts.negative}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Mitotic:</span>
                <span className="font-semibold text-purple-600">{cellCounts.mitotic}</span>
              </div>
              {totalManualCells > 0 && (
                <div className="pt-2 border-t border-gray-200">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Ki-67 Index:</span>
                    <span className="font-bold text-indigo-600">
                      {((cellCounts.positive / Math.max(totalManualCells - cellCounts.unclear, 1)) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Image Area */}
        <div className="flex-1">
          <div className="relative">
            {/* Status Banner */}
            {isCountingActive && (
              <div className="absolute top-4 left-4 z-10 bg-green-600 text-white px-4 py-2 rounded-lg shadow-lg">
                <div className="flex items-center">
                  <div className="w-2 h-2 bg-green-300 rounded-full mr-2 animate-pulse"></div>
                  <span className="font-medium">
                    Counting Mode: {cellTypes.find(t => t.id === countingMode)?.name}
                  </span>
                </div>
              </div>
            )}

            {/* Instructions */}
            <div className="absolute bottom-4 left-4 z-10 bg-black bg-opacity-75 text-white p-3 rounded-lg text-sm max-w-xs">
              {isCountingActive ? (
                <p>🖱️ Click on cells to mark them as {cellTypes.find(t => t.id === countingMode)?.name.toLowerCase()}</p>
              ) : (
                <p>📝 Click "Start Counting" to begin manual cell marking</p>
              )}
            </div>

            {/* Main Image with Overlay */}
            <div className="relative">
              <img
                src={originalImage?.preview}
                alt="Cell counting"
                className="w-full h-auto cursor-crosshair"
                onClick={handleImageClick}
                style={{ maxHeight: '600px', objectFit: 'contain' }}
              />
              
              {/* Manual Cell Overlays */}
              {showManualCells && (
                <svg 
                  className="absolute top-0 left-0 w-full h-full pointer-events-none"
                  viewBox="0 0 1024 1024"
                  preserveAspectRatio="xMidYMid meet"
                >
                  {manualCells.map((cell, index) => {
                    const cellType = cellTypes.find(t => t.id === cell.type);
                    return (
                      <g key={cell.id}>
                        <circle
                          cx={cell.x}
                          cy={cell.y}
                          r="12"
                          fill={cellType?.color}
                          fillOpacity="0.7"
                          stroke={cellType?.color}
                          strokeWidth="2"
                        />
                        <text
                          x={cell.x}
                          y={cell.y + 4}
                          textAnchor="middle"
                          fontSize="10"
                          fill="white"
                          fontWeight="bold"
                        >
                          {index + 1}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CellCountingTool;
