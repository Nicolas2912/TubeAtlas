import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Network, Database, Loader, AlertCircle, Download, RefreshCw } from 'lucide-react';
import KnowledgeGraphVisualization from '../components/features/KnowledgeGraphVisualization';
import KGService from '../services/kgService';
import LoadingSpinner from '../components/ui/LoadingSpinner';

const KnowledgeGraphPage = () => {
  const [graphData, setGraphData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedDataset, setSelectedDataset] = useState('bryanjohnson');
  const [useMockData, setUseMockData] = useState(true);
  const [selectedNode, setSelectedNode] = useState(null);
  const [selectedEdge, setSelectedEdge] = useState(null);

  // Available datasets
  const datasets = [
    { id: 'bryanjohnson', name: 'Bryan Johnson', description: 'Biohacking and longevity content' },
    { id: 'andreykarpathy', name: 'Andrej Karpathy', description: 'AI and machine learning content' },
  ];

  const loadKnowledgeGraph = async (dataset, useMock = false) => {
    console.log(`=== Loading KG: dataset=${dataset}, useMock=${useMock} ===`);
    setLoading(true);
    setError(null);

    try {
      let rawData;

      if (useMock) {
        console.log('Using mock data for testing');
        rawData = KGService.generateMockData();
        console.log('Mock data generated:', rawData);
      } else {
        console.log(`Loading knowledge graph for dataset: ${dataset}`);
        try {
          // Try to fetch from backend first
          rawData = await KGService.fetchKnowledgeGraph(dataset);
          console.log('Data fetched from backend:', rawData);
        } catch (backendError) {
          console.warn('Backend not available, trying to load local file:', backendError.message);
          // Fallback to local JSON file
          rawData = await KGService.loadLocalKG(`complete_kg_langchain_${dataset}.json`);
          console.log('Data loaded from local file:', rawData);
        }
      }

      if (!rawData || !rawData.triples || rawData.triples.length === 0) {
        throw new Error('No knowledge graph data found');
      }

      console.log(`Loaded ${rawData.triples.length} triples`);

      // Transform data for sigma.js
      const sigmaData = KGService.transformToSigmaFormat(rawData.triples);
      console.log(`Transformed to ${sigmaData.nodes.length} nodes and ${sigmaData.edges.length} edges`);
      console.log('Raw triples sample:', rawData.triples.slice(0, 3));
      console.log('Sigma data sample:', {
        nodes: sigmaData.nodes.slice(0, 3),
        edges: sigmaData.edges.slice(0, 3)
      });
      console.log('All edges created:', sigmaData.edges.map(e => `${e.source} --[${e.label}]--> ${e.target}`));

      setGraphData(sigmaData);
    } catch (err) {
      console.error('Error loading knowledge graph:', err);
      setError(err.message);

      // If there's an error and we weren't using mock data, try mock data as fallback
      if (!useMock) {
        console.log('Attempting fallback to mock data...');
        try {
          const mockData = KGService.generateMockData();
          const sigmaMockData = KGService.transformToSigmaFormat(mockData.triples);
          setGraphData(sigmaMockData);
          setError(null);
          console.log('Successfully loaded mock data as fallback');
        } catch (mockError) {
          console.error('Even mock data failed:', mockError);
        }
      }
    } finally {
      setLoading(false);
    }
  };

  // Load initial data
  useEffect(() => {
    loadKnowledgeGraph(selectedDataset, useMockData);
  }, [selectedDataset, useMockData]);

  const handleNodeClick = (nodeKey, nodeData) => {
    console.log('Node clicked:', nodeKey, nodeData);
    setSelectedNode({ key: nodeKey, ...nodeData });
  };

  const handleEdgeClick = (edgeKey, edgeData) => {
    console.log('Edge clicked:', edgeKey, edgeData);
    setSelectedEdge({ key: edgeKey, ...edgeData });
  };

  const handleRefresh = () => {
    loadKnowledgeGraph(selectedDataset, useMockData);
  };

  const handleDownload = () => {
    if (!graphData) return;

    const dataStr = JSON.stringify(graphData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `knowledge_graph_${selectedDataset}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white shadow-sm border-b"
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Network className="w-8 h-8 text-primary-600" />
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Knowledge Graph</h1>
                <p className="text-gray-600">Interactive visualization of extracted knowledge relationships</p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* Dataset Selector */}
              <div className="flex items-center space-x-2">
                <Database className="w-5 h-5 text-gray-500" />
                <select
                  value={selectedDataset}
                  onChange={(e) => setSelectedDataset(e.target.value)}
                  className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  disabled={loading}
                >
                  {datasets.map(dataset => (
                    <option key={dataset.id} value={dataset.id}>
                      {dataset.name}
                    </option>
                  ))}
                </select>
              </div>

              {/* Mock Data Toggle */}
              <label className="flex items-center space-x-2 text-sm">
                <input
                  type="checkbox"
                  checked={useMockData}
                  onChange={(e) => setUseMockData(e.target.checked)}
                  className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  disabled={loading}
                />
                <span className="text-gray-700">Use mock data</span>
              </label>

              {/* Action Buttons */}
              <div className="flex space-x-2">
                <button
                  onClick={handleRefresh}
                  disabled={loading}
                  className="p-2 bg-gray-100 hover:bg-gray-200 disabled:bg-gray-50 rounded-md transition-colors"
                  title="Refresh"
                >
                  <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
                </button>

                <button
                  onClick={handleDownload}
                  disabled={loading || !graphData}
                  className="p-2 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-400 text-white rounded-md transition-colors"
                  title="Download Graph Data"
                >
                  <Download className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Content */}
      <div className="flex-1 h-[calc(100vh-120px)]">
        {loading && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <LoadingSpinner size="lg" />
              <p className="mt-4 text-gray-600">Loading knowledge graph...</p>
            </div>
          </div>
        )}

        {error && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex items-center justify-center h-full"
          >
            <div className="text-center max-w-md">
              <AlertCircle className="w-16 h-16 text-red-500 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Error Loading Knowledge Graph</h3>
              <p className="text-gray-600 mb-4">{error}</p>
              <div className="space-y-2">
                <button
                  onClick={() => setUseMockData(true)}
                  className="w-full btn-primary"
                >
                  Try Mock Data
                </button>
                <button
                  onClick={handleRefresh}
                  className="w-full btn-secondary"
                >
                  Retry
                </button>
              </div>
            </div>
          </motion.div>
        )}

        {!loading && !error && graphData && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="h-full"
          >
            <KnowledgeGraphVisualization
              graphData={graphData}
              onNodeClick={handleNodeClick}
              onEdgeClick={handleEdgeClick}
              className="h-full"
            />
          </motion.div>
        )}

        {!loading && !error && !graphData && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <Database className="w-16 h-16 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-gray-900 mb-2">No Data Available</h3>
              <p className="text-gray-600 mb-4">The knowledge graph data could not be loaded.</p>
              <button
                onClick={() => {
                  setUseMockData(true);
                  loadKnowledgeGraph(selectedDataset, true);
                }}
                className="btn-primary"
              >
                Load Mock Data
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Selection Info Panel */}
      {(selectedNode || selectedEdge) && (
        <motion.div
          initial={{ opacity: 0, x: 300 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 300 }}
          className="fixed right-4 top-1/2 transform -translate-y-1/2 bg-white rounded-lg shadow-xl p-4 w-80 z-20"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold">
              {selectedNode ? 'Node Details' : 'Edge Details'}
            </h3>
            <button
              onClick={() => {
                setSelectedNode(null);
                setSelectedEdge(null);
              }}
              className="text-gray-400 hover:text-gray-600"
            >
              ×
            </button>
          </div>

          {selectedNode && (
            <div className="space-y-2 text-sm">
              <div><span className="font-medium">Label:</span> {selectedNode.label}</div>
              <div><span className="font-medium">Connections:</span> {selectedNode.size}</div>
              <div><span className="font-medium">Position:</span> ({selectedNode.x?.toFixed(2)}, {selectedNode.y?.toFixed(2)})</div>
              <div className="flex items-center">
                <span className="font-medium mr-2">Color:</span>
                <div
                  className="w-4 h-4 rounded-full"
                  style={{ backgroundColor: selectedNode.color }}
                />
              </div>
            </div>
          )}

          {selectedEdge && (
            <div className="space-y-2 text-sm">
              <div><span className="font-medium">Relationship:</span> {selectedEdge.label}</div>
              <div><span className="font-medium">Source:</span> {selectedEdge.source}</div>
              <div><span className="font-medium">Target:</span> {selectedEdge.target}</div>
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
};

export default KnowledgeGraphPage;
