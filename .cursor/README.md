# Cursor Configuration

This directory contains Cursor-specific configuration files for the TubeAtlas project.

## Files

- `mcp.json` - MCP (Model Context Protocol) server configuration for TaskMaster integration
- `mcp.json.example` - Template file showing the required structure
- `rules/` - Coding rules and guidelines for the project

## Security Warning ⚠️

**NEVER commit actual API keys to version control!**

The `mcp.json` file should only contain placeholder values in the repository. To set up your development environment:

1. Copy the structure from `mcp.json.example`
2. Replace placeholder values in `mcp.json` with your actual API keys
3. The file is already in the repository with placeholders - just update the values locally
4. **Never commit your actual API keys!**

## Required API Keys

For full functionality, you'll need API keys for:

- **OpenAI**: Required for knowledge graph generation and chat features
- **Anthropic**: Required for some AI operations
- **Perplexity**: Required for research-backed task operations
- **Google/YouTube**: Required for YouTube Data API access

Other API keys (XAI, OpenRouter, Mistral, Azure, Ollama) are optional depending on your configuration.

## TaskMaster Integration

The MCP configuration enables Cursor to use TaskMaster for project management. See the main README.md for TaskMaster usage instructions. 