import React, { useEffect, useRef, useState } from 'react';
import { Graph } from 'graphology';
import Sigma from 'sigma';
import { circular } from 'graphology-layout';
import forceAtlas2 from 'graphology-layout-forceatlas2';
import { Search, ZoomIn, ZoomOut, RotateCcw, Settings, Filter } from 'lucide-react';

const KnowledgeGraphVisualization = ({
  graphData,
  onNodeClick,
  onEdgeClick,
  className = ''
}) => {
  const containerRef = useRef(null);
  const sigmaRef = useRef(null);
  const graphRef = useRef(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedNode, setSelectedNode] = useState(null);
  const [nodeInfo, setNodeInfo] = useState(null);
  const [isLayoutRunning, setIsLayoutRunning] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    showLabels: true,
    nodeSize: 'degree',
    layout: 'force'
  });

  // Initialize sigma instance
  useEffect(() => {
    if (!containerRef.current || !graphData) {
      console.log('Container or graph data not ready:', {
        hasContainer: !!containerRef.current,
        hasGraphData: !!graphData,
        nodeCount: graphData?.nodes?.length,
        edgeCount: graphData?.edges?.length
      });
      return;
    }

    try {
      setIsLoading(true);
      setError(null);

      // Clean up previous instance
      if (sigmaRef.current) {
        sigmaRef.current.kill();
        sigmaRef.current = null;
      }

      // Create a new graph
      const graph = new Graph();
      graphRef.current = graph;

      console.log('Initializing graph with data:', graphData);

      // Add nodes first
      if (graphData.nodes && graphData.nodes.length > 0) {
        console.log(`Adding ${graphData.nodes.length} nodes to graph`);
        graphData.nodes.forEach((node, index) => {
          try {
            const nodeKey = String(node.key); // Ensure string key
            graph.addNode(nodeKey, {
              label: node.label || nodeKey,
              size: node.size || 8,
              color: node.color || '#6b7280',
              originalColor: node.color || '#6b7280',
              originalSize: node.size || 8,
              x: isFinite(node.x) ? Number(node.x) : 0,
              y: isFinite(node.y) ? Number(node.y) : 0
            });
            console.log(`✓ Added node ${index}: ${nodeKey}`);
          } catch (e) {
            console.error(`✗ Error adding node ${index}:`, e, node);
          }
        });
      }

      // Add edges after nodes
      if (graphData.edges && graphData.edges.length > 0) {
        console.log(`Adding ${graphData.edges.length} edges to graph`);
        console.log('First 3 raw edges from graphData:', graphData.edges.slice(0, 3));
        let addedEdges = 0;
        const usedEdgeKeys = new Set(); // Track used edge keys to avoid duplicates
        graphData.edges.forEach((edge, index) => {
          try {
            console.log(`Processing edge ${index}:`, edge);

            // Safely extract and convert keys to strings
            const sourceKey = edge.source ? String(edge.source) : null;
            const targetKey = edge.target ? String(edge.target) : null;

            console.log(`Edge ${index} keys: source="${sourceKey}", target="${targetKey}"`);

            // Validate that we have valid source and target
            if (!sourceKey || !targetKey) {
              console.warn(`✗ Edge ${index} has invalid source or target:`, { sourceKey, targetKey, edge });
              return;
            }

            console.log(`Nodes exist? source=${graph.hasNode(sourceKey)}, target=${graph.hasNode(targetKey)}`);

            if (graph.hasNode(sourceKey) && graph.hasNode(targetKey)) {
              // Ensure unique edge key
              let edgeKey = edge.key || `edge-${index}`;
              let keyCounter = 0;
              while (usedEdgeKeys.has(edgeKey)) {
                keyCounter++;
                edgeKey = `${edge.key || `edge-${index}`}-${keyCounter}`;
              }
              usedEdgeKeys.add(edgeKey);

              const edgeAttributes = {
                label: edge.label || edge.predicate || 'related_to',
                color: edge.color || '#666666',
                size: edge.size || 3,
                type: 'line',
                originalColor: edge.color || '#666666'
              };

              console.log(`Attempting to add edge:`, {
                key: edgeKey,
                source: sourceKey,
                target: targetKey,
                attributes: edgeAttributes
              });

              // Use a try-catch specifically for the addEdge call
              try {
                // Try the standard graphology signature: addEdge(source, target, attributes)
                // Some versions don't require the edge key as first parameter
                const newEdge = graph.addEdge(sourceKey, targetKey, {
                  ...edgeAttributes,
                  key: edgeKey // Include key in attributes
                });
                addedEdges++;
                console.log(`✓ Successfully added edge ${index}: ${sourceKey} --[${edge.label || edge.predicate}]--> ${targetKey} (key: ${newEdge})`);
              } catch (addEdgeError) {
                console.error(`✗ Failed to add edge ${index} with signature addEdge(source, target, attributes):`, addEdgeError);
                console.error('Attempted parameters:', { sourceKey, targetKey, edgeAttributes });

                // Try alternative approach with key as first parameter
                try {
                  const altKey = `alt-edge-${index}-${Date.now()}`;
                  graph.addEdge(altKey, sourceKey, targetKey, edgeAttributes);
                  addedEdges++;
                  console.log(`✓ Added edge with 4-parameter signature: ${altKey}`);
                } catch (altError) {
                  console.error(`✗ Both signatures failed:`, altError);

                  // Try with edge method that auto-generates key
                  try {
                    const autoEdge = graph.addEdge(sourceKey, targetKey);
                    // Set attributes separately
                    graph.setEdgeAttribute(autoEdge, 'label', edge.label || edge.predicate || 'related_to');
                    graph.setEdgeAttribute(autoEdge, 'color', edge.color || '#666666');
                    graph.setEdgeAttribute(autoEdge, 'size', edge.size || 3);
                    addedEdges++;
                    console.log(`✓ Added edge with auto-key and separate attributes: ${autoEdge}`);
                  } catch (finalError) {
                    console.error(`✗ All edge creation methods failed:`, finalError);
                  }
                }
              }
            } else {
              console.warn(`✗ Edge ${index} references non-existent nodes:`, {
                edge,
                sourceKey,
                targetKey,
                sourceExists: graph.hasNode(sourceKey),
                targetExists: graph.hasNode(targetKey),
                availableNodes: graph.nodes().slice(0, 5)
              });
            }
          } catch (e) {
            console.error(`✗ Error processing edge ${index}:`, e);
            console.error('Full edge object:', JSON.stringify(edge, null, 2));
          }
        });
        console.log(`Successfully added ${addedEdges}/${graphData.edges.length} edges`);
      }

      console.log(`Graph created with ${graph.order} nodes and ${graph.size} edges`);
      console.log('Sample edges from graph:', graph.mapEdges((edge, attributes) => ({ edge, attributes })).slice(0, 3));

      // Apply initial layout with better spacing
      if (filters.layout === 'circular') {
        circular.assign(graph);
        // Add some spacing to circular layout
        graph.forEachNode((node, attributes) => {
          const x = graph.getNodeAttribute(node, 'x');
          const y = graph.getNodeAttribute(node, 'y');
          graph.setNodeAttribute(node, 'x', x * 1.5);
          graph.setNodeAttribute(node, 'y', y * 1.5);
        });
      } else {
        // Better distributed positioning for large graphs
        const nodeCount = graph.order;
        const radius = Math.max(200, Math.sqrt(nodeCount) * 50); // Larger base radius for large graphs

        let nodeIndex = 0;
        graph.forEachNode((node, attributes) => {
          // Use improved spiral distribution for better spacing
          const goldenAngle = 2.39996322972865332; // Golden angle in radians
          const angle = nodeIndex * goldenAngle;
          const r = radius * Math.sqrt(nodeIndex / Math.max(nodeCount, 10)); // Prevent division by small numbers

          // Add some randomness to prevent perfect alignment
          const jitter = 20;
          const x = r * Math.cos(angle) + (Math.random() - 0.5) * jitter;
          const y = r * Math.sin(angle) + (Math.random() - 0.5) * jitter;

          graph.setNodeAttribute(node, 'x', x);
          graph.setNodeAttribute(node, 'y', y);
          nodeIndex++;
        });
      }

      // Validate all nodes have proper coordinates before creating Sigma
      graph.forEachNode((node, attributes) => {
        const x = graph.getNodeAttribute(node, 'x');
        const y = graph.getNodeAttribute(node, 'y');

        if (!isFinite(x) || !isFinite(y)) {
          console.warn(`Node ${node} has invalid coordinates: x=${x}, y=${y}. Setting to (0,0)`);
          graph.setNodeAttribute(node, 'x', 0);
          graph.setNodeAttribute(node, 'y', 0);
        }
      });

      console.log('Node coordinates validation complete. Sample nodes:');
      let sampleCount = 0;
      graph.forEachNode((node, attributes) => {
        if (sampleCount < 3) {
          console.log(`${node}: x=${graph.getNodeAttribute(node, 'x')}, y=${graph.getNodeAttribute(node, 'y')}`);
          sampleCount++;
        }
      });

      // Create sigma instance with enhanced readability settings
      const sigma = new Sigma(graph, containerRef.current, {
        // Node rendering - improved readability
        renderLabels: filters.showLabels,
        labelFont: '"Inter", "Segoe UI", "Roboto", "Helvetica Neue", sans-serif',
        labelSize: 16,
        labelWeight: '600',
        labelColor: { color: '#1f2937' },

        // Edge rendering - better visibility
        renderEdges: true,
        renderEdgeLabels: filters.showLabels,
        edgeLabelFont: '"Inter", "Segoe UI", "Roboto", "Helvetica Neue", sans-serif',
        edgeLabelSize: 12,
        edgeLabelWeight: '500',
        edgeLabelColor: { color: '#6b7280' },

        // Default colors - improved contrast
        defaultNodeColor: '#94a3b8',
        defaultEdgeColor: '#e2e8f0',

        // Performance and interaction settings
        hideEdgesOnMove: false,
        hideLabelsOnMove: false,
        enableEdgeClickEvents: true,
        enableEdgeWheelEvents: false,
        enableEdgeHoverEvents: true,

        // Reducers for enhanced styling - optimized for large graphs
        nodeReducer: (node, data) => {
          const nodeCount = graph.order;
          // Scale node sizes based on graph size for better readability
          const baseSize = nodeCount > 50 ? 8 : nodeCount > 25 ? 10 : 12;
          const maxSize = nodeCount > 50 ? 16 : nodeCount > 25 ? 20 : 24;

          return {
            ...data,
            size: Math.max(baseSize, Math.min(maxSize, data.size || baseSize + 2)),
            color: data.color || '#94a3b8',
            label: data.label || node,
            x: data.x || 0,
            y: data.y || 0,
            type: 'circle',
            borderColor: data.highlighted ? '#fbbf24' : '#ffffff',
            borderSize: data.highlighted ? 3 : 1,
            highlighted: data.highlighted || false
          };
        },
        edgeReducer: (edge, data) => {
          const nodeCount = graph.order;
          // Make edges thinner and more transparent for large graphs
          const edgeOpacity = nodeCount > 50 ? 0.4 : nodeCount > 25 ? 0.6 : 0.8;
          const baseSize = nodeCount > 50 ? 1 : nodeCount > 25 ? 1.5 : 2;

          return {
            ...data,
            color: data.color || `rgba(203, 213, 225, ${edgeOpacity})`,
            size: Math.max(baseSize, Math.min(3, data.size || baseSize)),
            label: data.label || '',
            type: 'line',
            curveness: 0.05,
            hidden: data.hidden || false
          };
        }
      });

      sigmaRef.current = sigma;

      // Event handlers
      sigma.on('clickNode', (event) => {
        const nodeKey = event.node;
        const nodeData = graph.getNodeAttributes(nodeKey);
        setSelectedNode(nodeKey);
        setNodeInfo(nodeData);
        onNodeClick?.(nodeKey, nodeData);
        console.log('Node clicked:', nodeKey, nodeData);
      });

      sigma.on('clickEdge', (event) => {
        const edgeKey = event.edge;
        const edgeData = graph.getEdgeAttributes(edgeKey);
        onEdgeClick?.(edgeKey, edgeData);
        console.log('Edge clicked:', edgeKey, edgeData);
      });

      sigma.on('clickStage', () => {
        setSelectedNode(null);
        setNodeInfo(null);
      });

      // Dynamic label visibility based on zoom level for large graphs
      if (graph.order > 25) {
        sigma.on('updated', () => {
          const camera = sigma.getCamera();
          const zoomRatio = camera.ratio;

          // Show labels only when zoomed in enough for large graphs
          const shouldShowLabels = zoomRatio < 0.8;
          const shouldShowEdgeLabels = zoomRatio < 0.5;

          sigma.setSetting('renderLabels', shouldShowLabels);
          sigma.setSetting('renderEdgeLabels', shouldShowEdgeLabels);
        });
      }

      // Force a refresh after initialization
      setTimeout(() => {
        if (sigma && !sigma.killed) {
          // Explicitly enable edge rendering
          sigma.setSetting('renderEdges', true);
          sigma.setSetting('renderEdgeLabels', graph.order <= 25); // Only show edge labels for smaller graphs initially
          sigma.setSetting('hideEdgesOnMove', graph.order > 50); // Hide edges on move for large graphs
          sigma.setSetting('hideLabelsOnMove', graph.order > 30); // Hide labels on move for medium+ graphs

          // Auto-fit the graph to view for better initial experience
          const camera = sigma.getCamera();
          const nodePositions = [];
          graph.forEachNode((node) => {
            nodePositions.push({
              x: graph.getNodeAttribute(node, 'x'),
              y: graph.getNodeAttribute(node, 'y')
            });
          });

          if (nodePositions.length > 0) {
            const minX = Math.min(...nodePositions.map(p => p.x));
            const maxX = Math.max(...nodePositions.map(p => p.x));
            const minY = Math.min(...nodePositions.map(p => p.y));
            const maxY = Math.max(...nodePositions.map(p => p.y));

            const width = maxX - minX;
            const height = maxY - minY;
            const centerX = (minX + maxX) / 2;
            const centerY = (minY + maxY) / 2;

            // Calculate zoom to fit all nodes with some padding
            const padding = 50;
            const containerWidth = containerRef.current.clientWidth;
            const containerHeight = containerRef.current.clientHeight;
            const zoomX = containerWidth / (width + padding * 2);
            const zoomY = containerHeight / (height + padding * 2);
            const zoom = Math.min(zoomX, zoomY, 1); // Don't zoom in more than 1:1

            camera.animate({
              x: centerX,
              y: centerY,
              ratio: 1 / zoom
            }, { duration: 1000 });
          }

          // Force multiple refreshes to ensure proper rendering
          sigma.refresh();
          setTimeout(() => {
            if (sigma && !sigma.killed) {
              sigma.refresh();
              console.log('Sigma settings after init:', {
                renderEdges: sigma.getSetting('renderEdges'),
                renderEdgeLabels: sigma.getSetting('renderEdgeLabels'),
                hideEdgesOnMove: sigma.getSetting('hideEdgesOnMove'),
                nodeCount: graph.order,
                edgeCount: graph.size
              });
            }
          }, 100);

          setIsLoading(false);
        }
      }, 100);

      console.log('Sigma instance created successfully');

    } catch (error) {
      console.error('Error initializing sigma:', error);
      setError(error.message);
      setIsLoading(false);
    }

    // Cleanup function
    return () => {
      if (sigmaRef.current && !sigmaRef.current.killed) {
        sigmaRef.current.kill();
        sigmaRef.current = null;
      }
    };
  }, [graphData, filters.layout, filters.showLabels]);

  // Handle search
  useEffect(() => {
    if (!sigmaRef.current || !graphRef.current || sigmaRef.current.killed) return;

    const graph = graphRef.current;
    const sigma = sigmaRef.current;

    try {
      if (searchTerm) {
        // Reset all nodes to original state
        graph.forEachNode((node, attributes) => {
          graph.setNodeAttribute(node, 'color', attributes.originalColor);
          graph.setNodeAttribute(node, 'size', attributes.originalSize);
          graph.setNodeAttribute(node, 'borderColor', '#ffffff');
          graph.setNodeAttribute(node, 'borderSize', 2);
          graph.setNodeAttribute(node, 'highlighted', false);
        });

        // Highlight matching nodes with better visual feedback
        const matchingNodes = [];
        graph.forEachNode((node, attributes) => {
          if (attributes.label.toLowerCase().includes(searchTerm.toLowerCase())) {
            matchingNodes.push(node);
            graph.setNodeAttribute(node, 'color', '#ef4444');
            graph.setNodeAttribute(node, 'size', (attributes.originalSize || 15) * 1.8);
            graph.setNodeAttribute(node, 'borderColor', '#fbbf24');
            graph.setNodeAttribute(node, 'borderSize', 4);
            graph.setNodeAttribute(node, 'highlighted', true);
          }
        });

        if (matchingNodes.length > 0) {
          // Focus on first matching node
          const nodePosition = sigma.getNodeDisplayData(matchingNodes[0]);
          if (nodePosition) {
            sigma.getCamera().animate(nodePosition, { duration: 1000 });
          }
        }
      } else {
        // Reset all nodes to original state when search is cleared
        graph.forEachNode((node, attributes) => {
          graph.setNodeAttribute(node, 'color', attributes.originalColor);
          graph.setNodeAttribute(node, 'size', attributes.originalSize);
          graph.setNodeAttribute(node, 'borderColor', '#ffffff');
          graph.setNodeAttribute(node, 'borderSize', 2);
          graph.setNodeAttribute(node, 'highlighted', false);
        });
      }

      sigma.refresh();
    } catch (error) {
      console.error('Error in search:', error);
    }
  }, [searchTerm]);

  // Control functions
  const zoomIn = () => {
    if (sigmaRef.current && !sigmaRef.current.killed) {
      const camera = sigmaRef.current.getCamera();
      camera.animatedZoom({ duration: 200 });
    }
  };

  const zoomOut = () => {
    if (sigmaRef.current && !sigmaRef.current.killed) {
      const camera = sigmaRef.current.getCamera();
      camera.animatedUnzoom({ duration: 200 });
    }
  };

  const resetView = () => {
    if (sigmaRef.current && !sigmaRef.current.killed) {
      const camera = sigmaRef.current.getCamera();
      camera.animate({ x: 0.5, y: 0.5, ratio: 1 }, { duration: 500 });
    }
  };

  const runForceLayout = () => {
    if (!graphRef.current || isLayoutRunning) return;

    try {
      setIsLayoutRunning(true);
      const graph = graphRef.current;

      // Configure and run enhanced force-directed layout optimized for large graphs
      const settings = forceAtlas2.inferSettings(graph);
      const nodeCount = graph.order;

      // Adjust settings based on graph size
      const iterations = nodeCount > 50 ? 150 : nodeCount > 25 ? 120 : 100;
      const scalingRatio = nodeCount > 50 ? 50 : nodeCount > 25 ? 30 : 10;
      const gravity = nodeCount > 50 ? 0.1 : nodeCount > 25 ? 0.2 : 0.3;

      forceAtlas2.assign(graph, {
        iterations,
        settings: {
          ...settings,
          adjustSizes: true,
          barnesHutOptimize: true,
          barnesHutTheta: 0.9, // Higher value for better performance with large graphs
          edgeWeightInfluence: 0.3, // Reduced for less edge clustering
          gravity: gravity,
          linLogMode: nodeCount > 30, // Use linLogMode for large graphs
          outboundAttractionDistribution: true,
          scalingRatio: scalingRatio, // Higher scaling for more separation
          slowDown: nodeCount > 50 ? 5 : 3, // More slowdown for stability
          strongGravityMode: false
        }
      });

      if (sigmaRef.current && !sigmaRef.current.killed) {
        sigmaRef.current.refresh();
      }

      setTimeout(() => setIsLayoutRunning(false), 1000);
    } catch (error) {
      console.error('Error running force layout:', error);
      setIsLayoutRunning(false);
    }
  };

  const getNodeTypeStats = () => {
    if (!graphData?.nodes) return {};

    const stats = {};
    graphData.nodes.forEach(node => {
      const color = node.color;
      if (!stats[color]) {
        stats[color] = { count: 0, label: getColorLabel(color) };
      }
      stats[color].count++;
    });

    return stats;
  };

  const getColorLabel = (color) => {
    const colorMap = {
      '#3b82f6': 'Research',
      '#ef4444': 'People',
      '#10b981': 'Technology',
      '#f59e0b': 'Health',
      '#8b5cf6': 'Organizations',
      '#6b7280': 'Other'
    };
    return colorMap[color] || 'Unknown';
  };

  const nodeStats = getNodeTypeStats();

  if (error) {
    return (
      <div className={`flex items-center justify-center w-full h-full ${className}`}>
        <div className="text-center">
          <p className="text-red-600 mb-2">Error initializing visualization:</p>
          <p className="text-sm text-gray-600">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`relative w-full h-full ${className}`}>
      {isLoading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-50 z-50">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600 mx-auto mb-2"></div>
            <p className="text-sm text-gray-600">Initializing visualization...</p>
          </div>
        </div>
      )}

      {/* Controls Panel */}
      <div className="absolute top-4 left-4 z-10 bg-white rounded-lg shadow-lg p-4 space-y-3">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
          <input
            type="text"
            placeholder="Search nodes..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 pr-4 py-2 border border-gray-300 rounded-md text-sm w-48 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
        </div>

        {/* Show large graph helper */}
        {graphData?.nodes?.length > 50 && (
          <div className="text-xs text-gray-600 bg-blue-50 p-2 rounded">
            💡 Large graph detected ({graphData.nodes.length} nodes)<br/>
            • Labels show when zoomed in<br/>
            • Use search to find specific nodes<br/>
            • Run layout to organize better
          </div>
        )}

        {/* Zoom Controls */}
        <div className="flex space-x-2">
          <button
            onClick={zoomIn}
            className="p-2 bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <button
            onClick={zoomOut}
            className="p-2 bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <button
            onClick={resetView}
            className="p-2 bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
            title="Reset View"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>

        {/* Layout Controls */}
        <button
          onClick={runForceLayout}
          disabled={isLayoutRunning}
          className="w-full flex items-center justify-center space-x-2 px-3 py-2 bg-primary-600 hover:bg-primary-700 disabled:bg-gray-400 text-white rounded-md transition-colors text-sm mb-2"
        >
          <Settings className="w-4 h-4" />
          <span>{isLayoutRunning ? 'Running...' : 'Run Layout'}</span>
        </button>


      </div>

      {/* Stats Panel */}
      <div className="absolute top-4 right-4 z-10 bg-white rounded-lg shadow-lg p-4">
        <h3 className="font-semibold text-sm mb-2 flex items-center">
          <Filter className="w-4 h-4 mr-2" />
          Node Types
        </h3>
        <div className="space-y-1">
          {Object.entries(nodeStats).map(([color, data]) => (
            <div key={color} className="flex items-center justify-between text-xs">
              <div className="flex items-center">
                <div
                  className="w-3 h-3 rounded-full mr-2"
                  style={{ backgroundColor: color }}
                />
                <span>{data.label}</span>
              </div>
              <span className="font-medium">{data.count}</span>
            </div>
          ))}
        </div>
        <div className="mt-3 pt-2 border-t text-xs text-gray-600">
          Total: {graphData?.nodes?.length || 0} nodes, {graphData?.edges?.length || 0} edges
        </div>
      </div>

      {/* Node Info Panel */}
      {nodeInfo && (
        <div className="absolute bottom-4 left-4 z-10 bg-white rounded-lg shadow-lg p-4 max-w-xs">
          <h3 className="font-semibold text-sm mb-2">Node Details</h3>
          <div className="space-y-1 text-xs">
            <div><span className="font-medium">Label:</span> {nodeInfo.label}</div>
            <div><span className="font-medium">Connections:</span> {nodeInfo.size}</div>
            <div><span className="font-medium">Type:</span> {getColorLabel(nodeInfo.color)}</div>
          </div>
        </div>
      )}

      {/* Sigma Container */}
      <div
        ref={containerRef}
        className="w-full h-full bg-gradient-to-br from-slate-50 to-gray-100 border border-gray-200 rounded-lg"
        style={{ minHeight: '400px' }}
      />
    </div>
  );
};

export default KnowledgeGraphVisualization;
