import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import RefinedDashboard from './pages/RefinedDashboard';
import RefinedImageAnalysis from './pages/RefinedImageAnalysis';
import Results from './pages/Results';
import ModelManagement from './pages/ModelManagement';
import ApiTest from './components/ApiTest';
import SimpleConnectionTest from './components/SimpleConnectionTest';
import LiveDataTest from './components/LiveDataTest';
import ImageDisplayTest from './components/ImageDisplayTest';
import './App.css';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <Router>
      <div className="min-h-screen bg-gray-50 flex flex-col">
        <Header onMenuClick={() => setSidebarOpen(!sidebarOpen)} />
        
        <div className="flex flex-1 pt-16 overflow-hidden">
          <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
          
          <main className="flex-1 overflow-auto bg-gray-50 min-w-0">
            <div className="min-h-full w-full">
              <div className="w-full px-4 sm:px-6 lg:px-8 xl:px-12 2xl:px-16 py-6 lg:py-8">
                <Routes>
                  <Route path="/" element={<RefinedDashboard />} />
                  <Route path="/analyze" element={<RefinedImageAnalysis />} />
                  <Route path="/results" element={<Results />} />
                  <Route path="/models" element={<ModelManagement />} />
                  <Route path="/api-test" element={<ApiTest />} />
                  <Route path="/simple-test" element={<SimpleConnectionTest />} />
                  <Route path="/live-test" element={<LiveDataTest />} />
                  <Route path="/image-test" element={<ImageDisplayTest />} />
                </Routes>
              </div>
            </div>
          </main>
        </div>
      </div>
    </Router>
  );
}

export default App;
