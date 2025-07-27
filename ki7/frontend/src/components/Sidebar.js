import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  HomeIcon,
  CameraIcon,
  ChartBarIcon,
  CogIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';

const Sidebar = ({ isOpen, onClose }) => {
  const navigation = [
    { name: 'Dashboard', href: '/', icon: HomeIcon },
    { name: 'Image Analysis', href: '/analyze', icon: CameraIcon },
    { name: 'Results & Reports', href: '/results', icon: ChartBarIcon },
    { name: 'Model Management', href: '/models', icon: CogIcon },
  ];

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-30 lg:hidden"
          onClick={onClose}
          style={{ top: '4rem' }}
        />
      )}
      
      {/* Sidebar */}
      <div className={`
        w-64 bg-white shadow-xl flex-shrink-0 transform transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        lg:translate-x-0
        fixed top-16 left-0 z-40 h-[calc(100vh-4rem)] lg:relative lg:top-0 lg:h-full lg:z-auto
        flex flex-col
      `}>
        <div className="flex items-center justify-between p-4 border-b border-gray-200 lg:hidden">
          <h2 className="text-lg font-semibold text-gray-900">Navigation</h2>
          <button
            onClick={onClose}
            className="p-2 rounded-md text-gray-600 hover:text-gray-900 hover:bg-gray-100"
          >
            <XMarkIcon className="h-6 w-6" />
          </button>
        </div>

        <nav className="flex-1 mt-0 lg:mt-8 px-4 py-4 overflow-y-auto">
          <ul className="space-y-2">
            {navigation.map((item) => (
              <li key={item.name}>
                <NavLink
                  to={item.href}
                  className={({ isActive }) =>
                    `flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors duration-200 w-full ${
                      isActive
                        ? 'bg-gradient-to-r from-primary-50 to-medical-50 text-primary-700 border-r-2 border-primary-500'
                        : 'text-gray-700 hover:bg-gray-100 hover:text-gray-900'
                    }`
                  }
                  onClick={() => window.innerWidth < 1024 && onClose()}
                >
                  <item.icon className="mr-3 h-5 w-5 flex-shrink-0" />
                  <span className="truncate">{item.name}</span>
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>

        <div className="p-4 flex-shrink-0">
          <div className="bg-gradient-to-r from-primary-500 to-medical-500 rounded-lg p-4 text-white">
            <h3 className="font-semibold text-sm">Need Help?</h3>
            <p className="text-xs mt-1 opacity-90">
              Check our documentation for detailed analysis guides.
            </p>
            <button className="mt-2 text-xs bg-white bg-opacity-20 px-3 py-1 rounded-full hover:bg-opacity-30 transition-colors">
              View Docs
            </button>
          </div>
        </div>
      </div>
    </>
  );
};

export default Sidebar;
