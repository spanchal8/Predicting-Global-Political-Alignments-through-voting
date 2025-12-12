#!/bin/bash
# Setup script for UN Voting ML Project
# This script creates a virtual environment and installs all dependencies

set -e  # Exit on error

echo "============================================================================"
echo "UN Voting ML Project - Environment Setup"
echo "============================================================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "============================================================================"
echo "Setup completed successfully!"
echo "============================================================================"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the project:"
echo "  python3 un_voting_ml_v2.py"
echo ""
echo "To deactivate the virtual environment:"
echo "  deactivate"
echo ""

