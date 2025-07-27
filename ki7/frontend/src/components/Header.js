import React from 'react';
import { Bars3Icon, BeakerIcon } from '@heroicons/react/24/outline';

const Header = ({ onMenuClick }) => {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200 fixed w-full top-0 z-50">
      <div className="flex items-center justify-between px-4 py-3">
        <div className="flex items-center">
          <button
            onClick={onMenuClick}
            className="lg:hidden p-2 rounded-md text-gray-600 hover:text-gray-900 hover:bg-gray-100 transition-colors"
          >
            <Bars3Icon className="h-6 w-6" />
          </button>
          
          <div className="flex items-center ml-4 lg:ml-0">
            <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-r from-medical-500 to-primary-600 rounded-lg">
              <BeakerIcon className="h-6 w-6 text-white" />
            </div>
            <div className="ml-3">
              <h1 className="text-xl font-bold text-gray-900">Ki-67 Analyzer Pro</h1>
              <p className="text-sm text-gray-500">Advanced Medical Image Analysis</p>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <div className="hidden md:flex items-center space-x-2 bg-green-50 px-3 py-1 rounded-full">
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-green-700 font-medium">Models Ready</span>
          </div>
          
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-gradient-to-r from-primary-500 to-medical-500 rounded-full flex items-center justify-center">
              <span className="text-white font-semibold text-sm">AI</span>
            </div>
            <div className="hidden sm:block">
              <p className="text-sm font-medium text-gray-900">AI Assistant</p>
              <p className="text-xs text-gray-500">Ready to analyze</p>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
