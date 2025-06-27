/**
 * Knowledge Graph API Service
 * Handles fetching and processing knowledge graph data from the backend
 */

const API_BASE_URL = 'http://localhost:8000'; // Adjust this to your backend URL

export class KGService {
  /**
   * Fetch knowledge graph data for a specific dataset
   * @param {string} dataset - The dataset name (e.g., 'bryanjohnson')
   * @returns {Promise<Object>} Knowledge graph data
   */
  static async fetchKnowledgeGraph(dataset = 'bryanjohnson') {
    try {
      const response = await fetch(`${API_BASE_URL}/api/kg/${dataset}`);

      if (!response.ok) {
        throw new Error(`Failed to fetch knowledge graph: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching knowledge graph:', error);
      throw error;
    }
  }

  /**
   * Load knowledge graph from local JSON file (for testing)
   * @param {string} filename - The JSON filename
   * @returns {Promise<Object>} Knowledge graph data
   */
  static async loadLocalKG(filename = 'complete_kg_langchain_bryanjohnson.json') {
    try {
      const response = await fetch(`/data/${filename}`);

      if (!response.ok) {
        throw new Error(`Failed to load local KG file: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error loading local knowledge graph:', error);
      throw error;
    }
  }

  /**
   * Transform triples data into sigma.js graph format
   * @param {Array} triples - Array of {subject, predicate, object} triples
   * @returns {Object} Graph data with nodes and edges for sigma.js
   */
  static transformToSigmaFormat(triples) {
    const nodes = new Map();
    const edges = [];

    // Extract unique nodes and create edges
    triples.forEach((triple, index) => {
      const { subject, predicate, object } = triple;

      // Ensure string keys
      const subjectKey = String(subject);
      const objectKey = String(object);

      // Add subject node
      if (!nodes.has(subjectKey)) {
        nodes.set(subjectKey, {
          key: subjectKey,
          label: subjectKey,
          size: 1,
          color: this.getNodeColor(subjectKey),
          x: Math.random() * 100,
          y: Math.random() * 100
        });
      }

      // Add object node
      if (!nodes.has(objectKey)) {
        nodes.set(objectKey, {
          key: objectKey,
          label: objectKey,
          size: 1,
          color: this.getNodeColor(objectKey),
          x: Math.random() * 100,
          y: Math.random() * 100
        });
      }

      // Add edge
      edges.push({
        key: `edge-${index}`,
        source: subjectKey,
        target: objectKey,
        label: predicate,
        predicate: predicate,  // Keep both for compatibility
        color: '#666666',
        size: 3,
        type: 'line'
      });
    });

    // Update node sizes based on connections
    const nodeDegrees = new Map();
    edges.forEach(edge => {
      const sourceKey = String(edge.source);
      const targetKey = String(edge.target);
      nodeDegrees.set(sourceKey, (nodeDegrees.get(sourceKey) || 0) + 1);
      nodeDegrees.set(targetKey, (nodeDegrees.get(targetKey) || 0) + 1);
    });

    // Scale node sizes based on degree
    Array.from(nodes.values()).forEach(node => {
      const nodeKey = String(node.key);
      const degree = nodeDegrees.get(nodeKey) || 1;
      node.size = Math.max(5, Math.min(25, degree * 3));
    });

    return {
      nodes: Array.from(nodes.values()),
      edges: edges
    };
  }

  /**
   * Get color for a node based on its type or content
   * @param {string} nodeLabel - The node label
   * @returns {string} Hex color code
   */
  static getNodeColor(nodeLabel) {
    const label = nodeLabel.toLowerCase();

    // Define color categories
    if (label.includes('research') || label.includes('study') || label.includes('paper')) {
      return '#3b82f6'; // Blue for research
    } else if (label.includes('person') || label.includes('author') || label.includes('researcher')) {
      return '#ef4444'; // Red for people
    } else if (label.includes('technology') || label.includes('ai') || label.includes('machine learning')) {
      return '#10b981'; // Green for technology
    } else if (label.includes('health') || label.includes('longevity') || label.includes('biohacking')) {
      return '#f59e0b'; // Orange for health
    } else if (label.includes('company') || label.includes('organization')) {
      return '#8b5cf6'; // Purple for organizations
    } else {
      return '#6b7280'; // Gray for others
    }
  }

  /**
   * Generate mock data for testing when backend is not available
   * @returns {Object} Mock knowledge graph data
   */
  static generateMockData() {
    const mockTriples = [
      { subject: "Bryan Johnson", predicate: "founded", object: "Kernel" },
      { subject: "Bryan Johnson", predicate: "practices", object: "Biohacking" },
      { subject: "Bryan Johnson", predicate: "follows", object: "Blueprint Protocol" },
      { subject: "Kernel", predicate: "develops", object: "Brain-Computer Interface" },
      { subject: "Blueprint Protocol", predicate: "focuses on", object: "Longevity" },
      { subject: "Blueprint Protocol", predicate: "includes", object: "Sleep Optimization" },
      { subject: "Blueprint Protocol", predicate: "includes", object: "Exercise Routine" },
      { subject: "Biohacking", predicate: "involves", object: "Health Optimization" },
      { subject: "Longevity", predicate: "requires", object: "Data Tracking" },
      { subject: "Sleep Optimization", predicate: "improves", object: "Cognitive Performance" },
      { subject: "Exercise Routine", predicate: "enhances", object: "Physical Health" },
      { subject: "Data Tracking", predicate: "enables", object: "Personalized Medicine" }
    ];

    return { triples: mockTriples };
  }
}

export default KGService;
