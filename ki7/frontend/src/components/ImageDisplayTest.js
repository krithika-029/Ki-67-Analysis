import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';

const ImageDisplayTest = () => {
  const [uploadedImage, setUploadedImage] = useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      const imageData = {
        file: file,
        preview: URL.createObjectURL(file),
        name: file.name,
        size: file.size,
        type: file.type
      };
      console.log('Image uploaded:', imageData);
      setUploadedImage(imageData);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.tiff', '.bmp']
    },
    maxFiles: 1,
  });

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h2 className="text-2xl font-bold mb-6">Image Display Test</h2>
      
      {/* Upload Area */}
      <div 
        {...getRootProps()} 
        className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer mb-8 ${
          isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-gray-400'
        }`}
      >
        <input {...getInputProps()} />
        <p className="text-lg text-gray-600">
          {isDragActive
            ? 'Drop the image here...'
            : 'Drag & drop an image here, or click to select'}
        </p>
      </div>

      {/* Debug Info */}
      {uploadedImage && (
        <div className="mb-8 p-4 bg-gray-100 rounded-lg">
          <h3 className="text-lg font-semibold mb-2">Debug Info:</h3>
          <p><strong>Name:</strong> {uploadedImage.name}</p>
          <p><strong>Size:</strong> {uploadedImage.size} bytes</p>
          <p><strong>Type:</strong> {uploadedImage.type}</p>
          <p><strong>Preview URL:</strong> {uploadedImage.preview}</p>
          <p><strong>Has preview:</strong> {uploadedImage.preview ? 'Yes' : 'No'}</p>
        </div>
      )}

      {/* Image Display Test */}
      {uploadedImage && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div className="bg-gray-50 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4">Method 1: Direct src</h3>
            <div className="border rounded-lg overflow-hidden bg-white">
              <img
                src={uploadedImage.preview}
                alt="Test 1"
                className="w-full h-auto"
                onLoad={() => console.log('Image loaded successfully')}
                onError={(e) => console.error('Image load error:', e)}
              />
            </div>
          </div>

          <div className="bg-gray-50 rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4">Method 2: Conditional render</h3>
            <div className="border rounded-lg overflow-hidden bg-white">
              {uploadedImage?.preview ? (
                <img
                  src={uploadedImage.preview}
                  alt="Test 2"
                  className="w-full h-auto"
                  onLoad={() => console.log('Conditional image loaded successfully')}
                  onError={(e) => console.error('Conditional image load error:', e)}
                />
              ) : (
                <div className="w-full h-48 bg-gray-200 flex items-center justify-center">
                  <span className="text-gray-500">No preview available</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageDisplayTest;
