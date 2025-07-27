import React, { useState, useRef, useEffect } from 'react';
import {
  MagnifyingGlassIcon,
  MagnifyingGlassPlusIcon,
  MagnifyingGlassMinusIcon,
  EyeIcon,
  EyeSlashIcon,
  AdjustmentsHorizontalIcon,
  DocumentArrowDownIcon,
  SwatchIcon
} from '@heroicons/react/24/outline';

const ImageAnnotationViewer = ({ originalImage, analysisResults }) => {
  const canvasRef = useRef(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [showPositiveCells, setShowPositiveCells] = useState(true);
  const [showNegativeCells, setShowNegativeCells] = useState(true);
  const [showROI, setShowROI] = useState(true);
  const [showHotSpots, setShowHotSpots] = useState(true);
  const [selectedCell, setSelectedCell] = useState(null);
  const [isPanning, setIsPanning] = useState(false);
  const [lastPanPoint, setLastPanPoint] = useState({ x: 0, y: 0 });

  const cellDetections = analysisResults?.cellDetections || [];
  const roiRegions = analysisResults?.roiRegions || [];
  const hotSpots = analysisResults?.hotSpots || [];

  useEffect(() => {
    drawAnnotations();
  }, [zoom, pan, showPositiveCells, showNegativeCells, showROI, showHotSpots, selectedCell]);

  const drawAnnotations = () => {
    const canvas = canvasRef.current;
    if (!canvas || !originalImage) return;

    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Save context state
      ctx.save();
      
      // Apply zoom and pan
      ctx.scale(zoom, zoom);
      ctx.translate(pan.x, pan.y);
      
      // Draw original image
      ctx.drawImage(img, 0, 0, 1024, 1024);
      
      // Draw ROI regions
      if (showROI) {
        roiRegions.forEach(roi => {
          ctx.strokeStyle = roi.proliferation_index > 30 ? '#ef4444' : roi.proliferation_index > 15 ? '#f59e0b' : '#10b981';
          ctx.lineWidth = 3 / zoom;
          ctx.setLineDash([10, 5]);
          ctx.strokeRect(roi.x, roi.y, roi.width, roi.height);
          
          // Draw ROI label
          ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
          ctx.fillRect(roi.x, roi.y - 25, roi.width, 25);
          ctx.fillStyle = 'white';
          ctx.font = `${16 / zoom}px Arial`;
          ctx.fillText(`${roi.annotation} (${roi.proliferation_index}%)`, roi.x + 5, roi.y - 8);
        });
        ctx.setLineDash([]);
      }

      // Draw hot spots
      if (showHotSpots) {
        hotSpots.forEach(spot => {
          const gradient = ctx.createRadialGradient(
            spot.x, spot.y, 0,
            spot.x, spot.y, spot.radius
          );
          gradient.addColorStop(0, 'rgba(255, 0, 0, 0.3)');
          gradient.addColorStop(1, 'rgba(255, 0, 0, 0.1)');
          
          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(spot.x, spot.y, spot.radius, 0, 2 * Math.PI);
          ctx.fill();
          
          // Hot spot border
          ctx.strokeStyle = spot.significance === 'high' ? '#dc2626' : '#f59e0b';
          ctx.lineWidth = 2 / zoom;
          ctx.setLineDash([5, 5]);
          ctx.stroke();
          ctx.setLineDash([]);
          
          // Hot spot label
          ctx.fillStyle = 'rgba(220, 38, 38, 0.9)';
          ctx.fillRect(spot.x - 40, spot.y - spot.radius - 20, 80, 18);
          ctx.fillStyle = 'white';
          ctx.font = `${12 / zoom}px Arial`;
          ctx.textAlign = 'center';
          ctx.fillText(`Hot Spot ${spot.ki67_percentage}%`, spot.x, spot.y - spot.radius - 6);
          ctx.textAlign = 'left';
        });
      }
      
      // Draw cell detections
      cellDetections.forEach(cell => {
        if (cell.type === 'positive' && !showPositiveCells) return;
        if (cell.type === 'negative' && !showNegativeCells) return;
        
        const isSelected = selectedCell && selectedCell.id === cell.id;
        const radius = Math.sqrt(cell.area / Math.PI);
        
        // Cell circle
        ctx.beginPath();
        ctx.arc(cell.x, cell.y, radius, 0, 2 * Math.PI);
        
        if (cell.type === 'positive') {
          const intensity = cell.intensity;
          if (intensity > 0.7) {
            ctx.fillStyle = 'rgba(220, 38, 38, 0.7)'; // Strong positive - bright red
            ctx.strokeStyle = '#dc2626';
          } else if (intensity > 0.4) {
            ctx.fillStyle = 'rgba(251, 146, 60, 0.7)'; // Moderate positive - orange
            ctx.strokeStyle = '#ea580c';
          } else {
            ctx.fillStyle = 'rgba(254, 240, 138, 0.7)'; // Weak positive - yellow
            ctx.strokeStyle = '#d97706';
          }
        } else {
          ctx.fillStyle = 'rgba(59, 130, 246, 0.5)'; // Negative - blue
          ctx.strokeStyle = '#2563eb';
        }
        
        ctx.fill();
        ctx.lineWidth = isSelected ? 3 / zoom : 1 / zoom;
        ctx.stroke();
        
        // Add confidence indicator for selected cell
        if (isSelected) {
          ctx.strokeStyle = '#000000';
          ctx.lineWidth = 2 / zoom;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.arc(cell.x, cell.y, radius + 5, 0, 2 * Math.PI);
          ctx.stroke();
          ctx.setLineDash([]);
        }
        
        // Cell ID for high-confidence cells or selected cell
        if (cell.confidence > 0.95 || isSelected) {
          ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
          ctx.font = `${10 / zoom}px Arial`;
          ctx.textAlign = 'center';
          ctx.fillText(cell.id.toString(), cell.x, cell.y + 3);
          ctx.textAlign = 'left';
        }
      });
      
      // Restore context state
      ctx.restore();
    };
    
    img.src = originalImage.preview;
  };

  const handleCanvasClick = (event) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left - pan.x * zoom) / zoom;
    const y = (event.clientY - rect.top - pan.y * zoom) / zoom;
    
    // Find clicked cell
    const clickedCell = cellDetections.find(cell => {
      const radius = Math.sqrt(cell.area / Math.PI);
      const distance = Math.sqrt((x - cell.x) ** 2 + (y - cell.y) ** 2);
      return distance <= radius;
    });
    
    setSelectedCell(clickedCell);
  };

  const handleMouseDown = (event) => {
    if (event.button === 0) { // Left mouse button
      setIsPanning(true);
      setLastPanPoint({ x: event.clientX, y: event.clientY });
    }
  };

  const handleMouseMove = (event) => {
    if (isPanning) {
      const deltaX = event.clientX - lastPanPoint.x;
      const deltaY = event.clientY - lastPanPoint.y;
      
      setPan(prev => ({
        x: prev.x + deltaX / zoom,
        y: prev.y + deltaY / zoom
      }));
      
      setLastPanPoint({ x: event.clientX, y: event.clientY });
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
  };

  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev * 1.2, 5));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev / 1.2, 0.1));
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const exportAnnotatedImage = () => {
    const canvas = canvasRef.current;
    const link = document.createElement('a');
    link.download = `ki67_annotated_${analysisResults?.filename || 'image'}.png`;
    link.href = canvas.toDataURL();
    link.click();
  };

  return (
    <div className="bg-white rounded-2xl shadow-lg overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-indigo-600 p-4 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold">Interactive Image Analysis</h3>
            <p className="text-purple-100 text-sm">
              {cellDetections.length} cells detected • Zoom: {(zoom * 100).toFixed(0)}%
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={handleZoomOut}
              className="p-2 bg-purple-700 hover:bg-purple-800 rounded-lg transition-colors"
              title="Zoom Out"
            >
              <MagnifyingGlassMinusIcon className="h-5 w-5" />
            </button>
            <button
              onClick={handleZoomIn}
              className="p-2 bg-purple-700 hover:bg-purple-800 rounded-lg transition-colors"
              title="Zoom In"
            >
              <MagnifyingGlassPlusIcon className="h-5 w-5" />
            </button>
            <button
              onClick={resetView}
              className="p-2 bg-purple-700 hover:bg-purple-800 rounded-lg transition-colors"
              title="Reset View"
            >
              <MagnifyingGlassIcon className="h-5 w-5" />
            </button>
            <button
              onClick={exportAnnotatedImage}
              className="p-2 bg-purple-700 hover:bg-purple-800 rounded-lg transition-colors"
              title="Export Annotated Image"
            >
              <DocumentArrowDownIcon className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>

      <div className="flex">
        {/* Controls Panel */}
        <div className="w-64 bg-gray-50 p-4 border-r border-gray-200">
          <h4 className="font-semibold text-gray-900 mb-4 flex items-center">
            <AdjustmentsHorizontalIcon className="h-5 w-5 mr-2" />
            Display Controls
          </h4>
          
          {/* Cell Type Toggles */}
          <div className="space-y-3 mb-6">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={showPositiveCells}
                onChange={(e) => setShowPositiveCells(e.target.checked)}
                className="rounded border-gray-300 text-red-600 focus:ring-red-500"
              />
              <div className="ml-3 flex items-center">
                <div className="w-4 h-4 bg-red-500 rounded-full mr-2"></div>
                <span className="text-sm font-medium">Positive Cells ({analysisResults?.positiveCells})</span>
              </div>
            </label>
            
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={showNegativeCells}
                onChange={(e) => setShowNegativeCells(e.target.checked)}
                className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
              />
              <div className="ml-3 flex items-center">
                <div className="w-4 h-4 bg-blue-500 rounded-full mr-2"></div>
                <span className="text-sm font-medium">Negative Cells ({analysisResults?.negativeCells})</span>
              </div>
            </label>
            
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={showROI}
                onChange={(e) => setShowROI(e.target.checked)}
                className="rounded border-gray-300 text-green-600 focus:ring-green-500"
              />
              <div className="ml-3 flex items-center">
                <div className="w-4 h-4 border-2 border-green-500 mr-2"></div>
                <span className="text-sm font-medium">ROI Regions ({roiRegions.length})</span>
              </div>
            </label>
            
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={showHotSpots}
                onChange={(e) => setShowHotSpots(e.target.checked)}
                className="rounded border-gray-300 text-yellow-600 focus:ring-yellow-500"
              />
              <div className="ml-3 flex items-center">
                <div className="w-4 h-4 bg-gradient-to-r from-red-500 to-yellow-500 rounded-full mr-2"></div>
                <span className="text-sm font-medium">Hot Spots ({hotSpots.length})</span>
              </div>
            </label>
          </div>

          {/* Legend */}
          <div className="border-t border-gray-200 pt-4">
            <h5 className="font-medium text-gray-900 mb-3 flex items-center">
              <SwatchIcon className="h-4 w-4 mr-2" />
              Color Legend
            </h5>
            <div className="space-y-2 text-xs">
              <div className="flex items-center">
                <div className="w-3 h-3 bg-red-600 rounded-full mr-2"></div>
                <span>Strong Positive (&gt;70%)</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 bg-orange-500 rounded-full mr-2"></div>
                <span>Moderate Positive (40-70%)</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 bg-yellow-400 rounded-full mr-2"></div>
                <span>Weak Positive (&lt;40%)</span>
              </div>
              <div className="flex items-center">
                <div className="w-3 h-3 bg-blue-500 rounded-full mr-2"></div>
                <span>Negative Cells</span>
              </div>
            </div>
          </div>

          {/* Selected Cell Info */}
          {selectedCell && (
            <div className="border-t border-gray-200 pt-4 mt-4">
              <h5 className="font-medium text-gray-900 mb-3">Cell Details</h5>
              <div className="space-y-2 text-xs">
                <p><span className="font-medium">ID:</span> {selectedCell.id}</p>
                <p><span className="font-medium">Type:</span> {selectedCell.type}</p>
                <p><span className="font-medium">Confidence:</span> {(selectedCell.confidence * 100).toFixed(1)}%</p>
                <p><span className="font-medium">Area:</span> {selectedCell.area} px²</p>
                <p><span className="font-medium">Intensity:</span> {(selectedCell.intensity * 100).toFixed(1)}%</p>
                <p><span className="font-medium">Shape Factor:</span> {selectedCell.shape_factor}</p>
                <p><span className="font-medium">Border:</span> {selectedCell.border_clarity}</p>
              </div>
            </div>
          )}
        </div>

        {/* Canvas Area */}
        <div className="flex-1 relative overflow-hidden">
          <canvas
            ref={canvasRef}
            width={1024}
            height={1024}
            className="border border-gray-200 cursor-crosshair"
            style={{
              width: '100%',
              height: '600px',
              objectFit: 'contain'
            }}
            onClick={handleCanvasClick}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          />
          
          {/* Instructions */}
          <div className="absolute bottom-4 left-4 bg-black bg-opacity-75 text-white p-3 rounded-lg text-sm">
            <p>🖱️ Click cells for details • Drag to pan • Use zoom controls</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageAnnotationViewer;
