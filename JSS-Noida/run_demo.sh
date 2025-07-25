#!/bin/bash

# Agentic Research Paper Analyzer - Demo Launcher
# Quick demonstration script for CS faculty

echo "🤖 Agentic Research Paper Analyzer - Demo Launcher"
echo "=================================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Setting up virtual environment..."
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    echo "✅ Setup complete!"
    echo ""
else
    source venv/bin/activate
fi

# Demo menu
echo "Select a demonstration:"
echo "1. Analyze Survey Paper (Deep Learning for Code Review)"
echo "2. Analyze Experimental Paper (Quantum ML for Cybersecurity)"
echo "3. Analyze User Study (VR Human-Computer Interaction)"
echo "4. Compare All Papers (Summary View)"
echo "5. JSON Output Example"
echo "6. Custom Paper Input"
echo ""

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        echo "🔍 Analyzing Survey Paper..."
        python -m src.cli --paper sample_data/paper1.txt
        ;;
    2)
        echo "🔍 Analyzing Experimental Paper..."
        python -m src.cli --paper sample_data/paper2.txt
        ;;
    3)
        echo "🔍 Analyzing User Study..."
        python -m src.cli --paper sample_data/paper3.txt
        ;;
    4)
        echo "📊 Comparing All Papers (Summary View)..."
        echo ""
        echo "=== PAPER 1: Survey Paper ==="
        python -m src.cli --paper sample_data/paper1.txt --output-format summary
        echo ""
        echo "=== PAPER 2: Experimental Paper ==="
        python -m src.cli --paper sample_data/paper2.txt --output-format summary
        echo ""
        echo "=== PAPER 3: User Study ==="
        python -m src.cli --paper sample_data/paper3.txt --output-format summary
        ;;
    5)
        echo "📄 JSON Output Example..."
        python -m src.cli --paper sample_data/paper1.txt --output-format json | head -30
        echo ""
        echo "(Output truncated - full JSON available with: python -m src.cli --paper sample_data/paper1.txt --output-format json)"
        ;;
    6)
        echo "📝 Enter path to your paper file:"
        read -p "File path: " filepath
        if [ -f "$filepath" ]; then
            python -m src.cli --paper "$filepath"
        else
            echo "❌ File not found: $filepath"
        fi
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        ;;
esac

echo ""
echo "🎓 Demo complete! For more options, see: python -m src.cli --help"
