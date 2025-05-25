"""
Knowledge Graph Visualization Module.

This module provides functionality to visualize knowledge graphs using different visualization
libraries (Plotly and PyVis). It supports interactive HTML-based visualizations with customizable
styling and layout options.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

import plotly.graph_objects as go
import networkx as nx
from pyvis.network import Network

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class VisualizationConfig:
    """Configuration class for knowledge graph visualization settings."""
    
    height: int = 800
    width: int = 1200
    node_size: int = 30
    node_color: str = '#1f77b4'
    node_border_color: str = '#ffffff'
    edge_color: str = '#888'
    edge_width: float = 1.5
    background_color: str = 'white'
    font_color: str = 'white'
    physics_enabled: bool = True
    spring_layout_k: float = 2.0
    spring_layout_iterations: int = 100
    seed: int = 42

class KnowledgeGraphVisualizer:
    """
    A class for visualizing knowledge graphs using different visualization libraries.
    
    This class provides methods to visualize knowledge graphs using either Plotly or PyVis,
    supporting interactive HTML-based visualizations with customizable styling options.
    
    Attributes:
        config (VisualizationConfig): Configuration settings for visualization
        graph (nx.DiGraph): NetworkX directed graph representation of the knowledge graph
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize the KnowledgeGraphVisualizer.
        
        Args:
            config: Optional configuration settings for visualization.
                   If None, default settings will be used.
        """
        self.config = config or VisualizationConfig()
        self.graph = nx.DiGraph()
        logger.info("Initialized KnowledgeGraphVisualizer with %s", 
                   "custom config" if config else "default config")

    def load_knowledge_graph(self, kg_file_path: Union[str, Path]) -> None:
        """
        Load a knowledge graph from a JSON file.
        
        Args:
            kg_file_path: Path to the JSON file containing the knowledge graph.
            
        Raises:
            FileNotFoundError: If the specified file does not exist.
            json.JSONDecodeError: If the file contains invalid JSON.
            KeyError: If the JSON structure is invalid (missing 'triples' key).
        """
        try:
            with open(kg_file_path, "r") as f:
                kg_data = json.load(f)
                kg = kg_data.get('triples', [])
                
            if not kg:
                logger.warning("No triples found in the knowledge graph file")
                return
                
            logger.info("Loaded knowledge graph with %d triples", len(kg))
            
            # Clear existing graph
            self.graph.clear()
            
            # Add nodes and edges
            for item in kg:
                if isinstance(item, dict):
                    source = item.get('subject')
                    target = item.get('object')
                    relation = item.get('predicate')
                    
                    if all([source, target, relation]):
                        self.graph.add_node(source, type='entity')
                        self.graph.add_node(target, type='entity')
                        self.graph.add_edge(source, target, relation=relation)
                    else:
                        logger.warning("Skipping invalid triple: %s", item)
                        
        except FileNotFoundError:
            logger.error("Knowledge graph file not found: %s", kg_file_path)
            raise
        except json.JSONDecodeError as e:
            logger.error("Invalid JSON in knowledge graph file: %s", e)
            raise
        except Exception as e:
            logger.error("Error loading knowledge graph: %s", e)
            raise

    def visualize_with_pyvis(self, output_file: Union[str, Path]) -> None:
        """
        Visualize the graph using PyVis and save to an HTML file.
        
        Args:
            output_file: Path where the HTML visualization will be saved.
            
        Raises:
            ValueError: If the graph is empty.
        """
        if not self.graph.nodes():
            logger.error("Cannot visualize empty graph")
            raise ValueError("Graph is empty. Load a knowledge graph first.")
            
        try:
            net = Network(
                notebook=True,
                height=f"{self.config.height}px",
                width="100%",
                bgcolor="#222222",
                font_color=self.config.font_color
            )
            
            # Add nodes and edges
            for node in self.graph.nodes():
                net.add_node(node, label=str(node))
                
            for edge in self.graph.edges(data=True):
                net.add_edge(
                    edge[0],
                    edge[1],
                    label=edge[2].get('relation', '')
                )
            
            # Configure visualization
            net.toggle_physics(self.config.physics_enabled)
            net.show_buttons(filter_=['physics'])
            
            # Save visualization
            output_path = Path(output_file)
            net.show(str(output_path))
            logger.info("PyVis visualization saved to %s", output_path)
            
        except Exception as e:
            logger.error("Error creating PyVis visualization: %s", e)
            raise

    def visualize_with_plotly(self, output_file: Union[str, Path]) -> None:
        """
        Visualize the graph using Plotly and save to an HTML file.
        
        Args:
            output_file: Path where the HTML visualization will be saved.
            
        Raises:
            ValueError: If the graph is empty.
        """
        if not self.graph.nodes():
            logger.error("Cannot visualize empty graph")
            raise ValueError("Graph is empty. Load a knowledge graph first.")
            
        try:
            # Calculate node positions
            pos = nx.spring_layout(
                self.graph,
                k=self.config.spring_layout_k,
                iterations=self.config.spring_layout_iterations,
                seed=self.config.seed
            )
            
            # Create edge traces
            edge_trace, edge_label_trace = self._create_edge_traces(pos)
            
            # Create node trace
            node_trace = self._create_node_trace(pos)
            
            # Create and save figure
            fig = self._create_figure(edge_trace, edge_label_trace, node_trace)
            output_path = Path(output_file)
            fig.write_html(str(output_path))
            logger.info("Plotly visualization saved to %s", output_path)
            
        except Exception as e:
            logger.error("Error creating Plotly visualization: %s", e)
            raise

    def _create_edge_traces(self, pos: Dict[str, tuple]) -> tuple:
        """
        Create edge traces for Plotly visualization.
        
        Args:
            pos: Dictionary of node positions.
            
        Returns:
            Tuple containing edge trace and edge label trace.
        """
        edge_x, edge_y = [], []
        edge_text = []
        edge_labels = []
        
        for edge in self.graph.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_text.append(f"{edge[0]} → {edge[2]['relation']} → {edge[1]}")
            edge_labels.append({
                'x': (x0 + x1) / 2,
                'y': (y0 + y1) / 2,
                'text': edge[2]['relation']
            })
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=self.config.edge_width, color=self.config.edge_color),
            hoverinfo='text',
            mode='lines',
            text=edge_text
        )
        
        edge_label_trace = go.Scatter(
            x=[label['x'] for label in edge_labels],
            y=[label['y'] for label in edge_labels],
            mode='text',
            text=[label['text'] for label in edge_labels],
            textposition='middle center',
            hoverinfo='none',
            textfont=dict(size=10, color='#666')
        )
        
        return edge_trace, edge_label_trace

    def _create_node_trace(self, pos: Dict[str, tuple]) -> go.Scatter:
        """
        Create node trace for Plotly visualization.
        
        Args:
            pos: Dictionary of node positions.
            
        Returns:
            Plotly Scatter object for nodes.
        """
        node_x = []
        node_y = []
        node_text = []
        
        for node in self.graph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(str(node))
        
        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=self.config.node_size,
                color=self.config.node_color,
                line=dict(width=2, color=self.config.node_border_color)
            )
        )

    def _create_figure(self, edge_trace: go.Scatter, 
                      edge_label_trace: go.Scatter,
                      node_trace: go.Scatter) -> go.Figure:
        """
        Create the final Plotly figure.
        
        Args:
            edge_trace: Edge trace for the graph.
            edge_label_trace: Edge label trace for the graph.
            node_trace: Node trace for the graph.
            
        Returns:
            Plotly Figure object.
        """
        return go.Figure(
            data=[edge_trace, edge_label_trace, node_trace],
            layout=go.Layout(
                title='Knowledge Graph Visualization',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=self.config.height,
                width=self.config.width,
                plot_bgcolor=self.config.background_color
            )
        )

def main():
    """Example usage of the KnowledgeGraphVisualizer class."""
    try:
        # Create visualizer with default config
        visualizer = KnowledgeGraphVisualizer()
        
        # Load and visualize knowledge graph
        kg_path = "data/example_kg_langchain.json"
        output_file = "data/kg_visualization_langchain.html"
        
        visualizer.load_knowledge_graph(kg_path)
        visualizer.visualize_with_pyvis(output_file)
        
    except Exception as e:
        logger.error("Error in main execution: %s", e)
        raise

if __name__ == "__main__":
    main()