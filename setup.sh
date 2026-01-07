#!/bin/bash
# Setup script for AI Notes Q&A Assistant

echo "🚀 Setting up AI Notes Q&A Assistant..."

# Check if python3-venv is installed
if ! python3 -m venv --help &>/dev/null; then
    echo "❌ python3-venv is not installed."
    echo "Please install it with: sudo apt install python3-venv"
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the app:"
echo "  1. Activate the virtual environment: source venv/bin/activate"
echo "  2. Run: streamlit run app/main.py"
echo ""

