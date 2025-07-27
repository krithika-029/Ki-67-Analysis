import React, { useState, useEffect } from 'react';
import {
  MagnifyingGlassIcon,
  CalendarIcon,
  FunnelIcon,
  ArrowDownTrayIcon,
  EyeIcon,
  ChartBarIcon,
} from '@heroicons/react/24/outline';
import { getAnalysisHistory } from '../services/api';

const Results = () => {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [sortBy, setSortBy] = useState('date');
  const [filterBy, setFilterBy] = useState('all');

  useEffect(() => {
    fetchResults();
  }, []);

  const fetchResults = async () => {
    try {
      const data = await getAnalysisHistory();
      setResults(data);
    } catch (error) {
      console.error('Failed to fetch results:', error);
    } finally {
      setLoading(false);
    }
  };

  const filteredResults = results
    .filter(result => {
      const matchesSearch = result.filename.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesFilter = filterBy === 'all' || 
        (filterBy === 'high' && result.ki67Index > 20) ||
        (filterBy === 'medium' && result.ki67Index >= 10 && result.ki67Index <= 20) ||
        (filterBy === 'low' && result.ki67Index < 10);
      return matchesSearch && matchesFilter;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'date':
          return new Date(b.date) - new Date(a.date);
        case 'filename':
          return a.filename.localeCompare(b.filename);
        case 'ki67Index':
          return b.ki67Index - a.ki67Index;
        case 'confidence':
          return b.confidence - a.confidence;
        default:
          return 0;
      }
    });

  const getKi67Category = (index) => {
    if (index > 20) return { label: 'High', color: 'text-red-600 bg-red-50' };
    if (index >= 10) return { label: 'Medium', color: 'text-yellow-600 bg-yellow-50' };
    return { label: 'Low', color: 'text-green-600 bg-green-50' };
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const handleExportAll = () => {
    // Implement CSV/Excel export
    console.log('Exporting all results...');
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900">Analysis Results</h1>
            <p className="text-gray-600 mt-1">
              View and manage your Ki-67 analysis history
            </p>
          </div>
          <button
            onClick={handleExportAll}
            className="mt-4 sm:mt-0 inline-flex items-center px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
          >
            <ArrowDownTrayIcon className="h-4 w-4 mr-2" />
            Export All
          </button>
        </div>

        {/* Stats Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">
          <div className="bg-blue-50 rounded-lg p-4">
            <p className="text-blue-600 text-sm font-medium">Total Analyses</p>
            <p className="text-2xl font-bold text-blue-900">{results.length}</p>
          </div>
          <div className="bg-green-50 rounded-lg p-4">
            <p className="text-green-600 text-sm font-medium">Avg Ki-67 Index</p>
            <p className="text-2xl font-bold text-green-900">
              {results.length > 0 ? 
                (results.reduce((sum, r) => sum + r.ki67Index, 0) / results.length).toFixed(1) 
                : '0'
              }%
            </p>
          </div>
          <div className="bg-purple-50 rounded-lg p-4">
            <p className="text-purple-600 text-sm font-medium">Avg Confidence</p>
            <p className="text-2xl font-bold text-purple-900">
              {results.length > 0 ? 
                (results.reduce((sum, r) => sum + r.confidence, 0) / results.length).toFixed(1) 
                : '0'
              }%
            </p>
          </div>
          <div className="bg-orange-50 rounded-lg p-4">
            <p className="text-orange-600 text-sm font-medium">This Month</p>
            <p className="text-2xl font-bold text-orange-900">
              {results.filter(r => 
                new Date(r.date).getMonth() === new Date().getMonth()
              ).length}
            </p>
          </div>
        </div>
      </div>

      {/* Filters and Search */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0 lg:space-x-4">
          {/* Search */}
          <div className="relative flex-1">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="text"
              placeholder="Search by filename..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 pr-4 py-2 w-full border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            />
          </div>

          {/* Sort */}
          <div className="flex items-center space-x-2">
            <CalendarIcon className="h-5 w-5 text-gray-400" />
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="date">Sort by Date</option>
              <option value="filename">Sort by Filename</option>
              <option value="ki67Index">Sort by Ki-67 Index</option>
              <option value="confidence">Sort by Confidence</option>
            </select>
          </div>

          {/* Filter */}
          <div className="flex items-center space-x-2">
            <FunnelIcon className="h-5 w-5 text-gray-400" />
            <select
              value={filterBy}
              onChange={(e) => setFilterBy(e.target.value)}
              className="border border-gray-300 rounded-lg px-3 py-2 focus:ring-2 focus:ring-primary-500 focus:border-primary-500"
            >
              <option value="all">All Results</option>
              <option value="high">High Ki-67 (&gt;20%)</option>
              <option value="medium">Medium Ki-67 (10-20%)</option>
              <option value="low">Low Ki-67 (&lt;10%)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Results Table */}
      <div className="bg-white rounded-xl shadow-lg overflow-hidden">
        {filteredResults.length === 0 ? (
          <div className="text-center py-12">
            <ChartBarIcon className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No results found</h3>
            <p className="text-gray-600">
              {searchTerm || filterBy !== 'all' 
                ? 'Try adjusting your search or filter criteria.'
                : 'Start analyzing images to see results here.'
              }
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Image
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Ki-67 Index
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Confidence
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Category
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Date
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {filteredResults.map((result) => {
                  const category = getKi67Category(result.ki67Index);
                  return (
                    <tr key={result.id} className="hover:bg-gray-50 transition-colors">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="flex-shrink-0 h-10 w-10">
                            <div className="h-10 w-10 rounded-lg bg-gradient-to-r from-primary-500 to-medical-500 flex items-center justify-center">
                              <span className="text-white font-semibold text-sm">
                                {result.filename.charAt(0).toUpperCase()}
                              </span>
                            </div>
                          </div>
                          <div className="ml-4">
                            <div className="text-sm font-medium text-gray-900">
                              {result.filename}
                            </div>
                            <div className="text-sm text-gray-500">
                              ID: {result.id}
                            </div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-semibold text-gray-900">
                          {result.ki67Index}%
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className="text-sm text-gray-900">
                            {result.confidence}%
                          </div>
                          <div className="ml-2 w-16 bg-gray-200 rounded-full h-2">
                            <div 
                              className="bg-green-500 h-2 rounded-full" 
                              style={{ width: `${result.confidence}%` }}
                            ></div>
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${category.color}`}>
                          {category.label}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatDate(result.date)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <div className="flex space-x-2">
                          <button className="text-primary-600 hover:text-primary-900">
                            <EyeIcon className="h-4 w-4" />
                          </button>
                          <button className="text-gray-400 hover:text-gray-600">
                            <ArrowDownTrayIcon className="h-4 w-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default Results;
