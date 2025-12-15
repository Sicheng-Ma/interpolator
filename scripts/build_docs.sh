#!/bin/bash
# =============================================================================
# Build Documentation Script
# =============================================================================
# This script builds the Sphinx documentation for the 5D Regressor project.
# 
# Usage:
#   ./scripts/build_docs.sh
#
# Output:
#   HTML documentation in docs/build/html/
# =============================================================================

set -e  # Exit on error

echo "========================================"
echo "Building 5D Regressor Documentation"
echo "========================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Navigate to project root
cd "$PROJECT_ROOT"

echo ""
echo "Step 1: Installing dependencies..."
echo "----------------------------------------"

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed"
    exit 1
fi

# Install Sphinx and dependencies
pip install sphinx sphinx-rtd-theme myst-parser --quiet

echo "Dependencies installed."

echo ""
echo "Step 2: Installing fivedreg package..."
echo "----------------------------------------"

# Install the backend package (needed for autodoc)
cd "$PROJECT_ROOT/backend"
pip install -e . --quiet

echo "Package installed."

echo ""
echo "Step 3: Building HTML documentation..."
echo "----------------------------------------"

# Navigate to docs directory
cd "$PROJECT_ROOT/docs"

# Clean previous build
rm -rf build/

# Build HTML
sphinx-build -b html source build/html

echo ""
echo "========================================"
echo "Documentation built successfully!"
echo "========================================"
echo ""
echo "Location: $PROJECT_ROOT/docs/build/html/index.html"
echo ""

# Try to open in browser
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open "build/html/index.html"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v xdg-open &> /dev/null; then
        xdg-open "build/html/index.html"
    fi
fi

echo "You can view the documentation by opening:"
echo "  file://$PROJECT_ROOT/docs/build/html/index.html"