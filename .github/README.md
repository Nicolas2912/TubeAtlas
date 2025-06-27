# TubeAtlas CI/CD Pipeline

This directory contains the GitHub Actions workflow configuration for TubeAtlas.

## Workflow Overview

The CI/CD pipeline includes four main jobs:

1. **Lint** - Code quality checks using pre-commit hooks
2. **Test** - Run pytest with coverage reporting
3. **Build** - Build and push Docker images to GitHub Container Registry
4. **Security Scan** - Vulnerability scanning with Trivy

## Required Secrets

No additional secrets are required for basic operation. The pipeline uses:

- `GITHUB_TOKEN` (automatically provided by GitHub) for:
  - Authenticating to GitHub Container Registry (ghcr.io)
  - Uploading security scan results

## Coverage Reporting

Coverage reports are automatically generated and available as artifacts:

- **XML Report**: For integration with other tools if needed
- **HTML Report**: Interactive coverage report for detailed analysis
- **Terminal Output**: Summary displayed during test execution

Coverage artifacts are retained for 30 days and can be downloaded from the Actions tab.

## Container Registry

Docker images are automatically pushed to:
- `ghcr.io/owner/repository-name:latest` (main branch)
- `ghcr.io/owner/repository-name:sha-{commit}` (all pushes)
- `ghcr.io/owner/repository-name:pr-{number}` (pull requests)

## Branch Protection

It's recommended to enable branch protection rules for the `main` branch:

1. Go to Settings → Branches in your repository
2. Add protection rule for `main`
3. Require status checks to pass before merging:
   - `lint`
   - `test`
   - `build`

## Local Development

To run the same checks locally:

```bash
# Install pre-commit hooks
poetry run pre-commit install

# Run all checks
poetry run pre-commit run --all-files

# Run tests with coverage
poetry run pytest --cov=src/tubeatlas --cov-report=html

# Build Docker image
docker compose build
```

## Troubleshooting

### Pre-commit Hook Failures

If pre-commit hooks fail, run locally first:
```bash
poetry run pre-commit run --all-files
```

### Test Failures

Run tests locally with verbose output:
```bash
poetry run pytest -v --cov=src/tubeatlas
```

### Docker Build Issues

Test Docker build locally:
```bash
docker compose build --no-cache
```
