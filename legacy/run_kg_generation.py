#!/usr/bin/env python3
"""
Script to run knowledge graph generation for testing purposes.

This script provides a simple way to generate knowledge graphs from
the existing data for testing the visualization system.
"""

import asyncio
import sys
from pathlib import Path

# Add the tubeatlas module to the path
sys.path.append(str(Path(__file__).parent))

from tubeatlas.kg_builder_langchain import GraphBuilderConfig, KnowledgeGraphBuilder


def main():
    """Generate knowledge graph for Bryan Johnson dataset."""
    try:
        print("🚀 Starting knowledge graph generation for Bryan Johnson dataset...")

        # Create configuration
        config = GraphBuilderConfig(
            model_name="gpt-4.1-mini",  # Use the correct model name
            temperature=0.0,
            strict_mode=True,
            db_path="data/bryanjohnson.db",
            output_path="data/complete_kg_langchain_bryanjohnson.json",
            additional_instructions="Focus on direct factual relationships. Identify specific research topics, health practices, biohacking techniques, and business relationships.",
        )

        print(f"📊 Configuration:")
        print(f"  - Model: {config.model_name}")
        print(f"  - Database: {config.db_path}")
        print(f"  - Output: {config.output_path}")
        print()

        # Check if database exists
        if not Path(config.db_path).exists():
            print(f"❌ Database file not found: {config.db_path}")
            print("Please ensure you have the Bryan Johnson data available.")
            return

        # Create builder
        builder = KnowledgeGraphBuilder(config)

        print("💰 Calculating estimated API costs...")
        total_tokens, estimated_cost = builder._calc_api_costs()
        print(f"  - Total tokens: {total_tokens:,}")
        print(f"  - Estimated cost: ${estimated_cost}")
        print()

        print("🔄 Starting knowledge graph generation...")

        # Option 1: Use synchronous version for smaller datasets
        print("📝 Processing transcripts (synchronous mode)...")
        graph_data = builder.build_complete_knowledge_graph(batch_size=10)

        # Option 2: Use async version for better performance (uncomment to use)
        # print("📝 Processing transcripts (async mode)...")
        # graph_data = asyncio.run(builder.build_complete_knowledge_graph_async(batch_size=10, max_concurrent=3))

        # Save the results
        print("💾 Saving knowledge graph...")
        builder.save_knowledge_graph(graph_data)

        # Print statistics
        triple_count = len(graph_data.get("triples", []))
        print()
        print("✅ Knowledge graph generation completed!")
        print(f"📈 Generated {triple_count:,} knowledge triples")
        print(f"📁 Saved to: {config.output_path}")

        # Verify the output file
        output_path = Path(config.output_path)
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"📏 File size: {file_size:,} bytes ({file_size / 1024:.1f} KB)")

        print()
        print(
            "🎉 You can now start the API server and frontend to visualize the knowledge graph!"
        )
        print("   - Start API: python api_server.py")
        print("   - Start frontend: cd frontend && npm run dev")

    except KeyboardInterrupt:
        print("\n❌ Operation interrupted by user.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
