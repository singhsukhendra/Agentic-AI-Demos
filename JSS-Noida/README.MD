# AI-Powered Agentic Research Paper Analyzer

Developed by Dr Mahesha BR Pandit

An advanced research paper analysis system that demonstrates the **Sense-Plan-Act** AI cycle using Large Language Models (LLMs) for actual reasoning and decision-making. This is the AI-powered evolution of the rule-based research analyzer, showcasing real AI reasoning capabilities for JSS-Noida CSE faculty demonstrations.

## 🚀 Key Features

### AI-Powered Analysis
- **LLM Integration**: Uses OpenAI, Anthropic, or local models for actual AI reasoning
- **Intelligent Sense Phase**: LLM analyzes paper content and extracts deep insights
- **Strategic Plan Phase**: LLM creates tailored analysis strategies based on paper characteristics
- **Comprehensive Act Phase**: LLM executes detailed analysis with reasoning transparency

### Multi-Provider Support
- **OpenAI**: GPT-3.5-turbo, GPT-4, GPT-4-turbo
- **Anthropic**: Claude-3 (Sonnet, Opus, Haiku)
- **Local Models**: Ollama, LocalAI, or custom endpoints
- **Automatic Fallback**: Graceful degradation between providers

### Advanced Capabilities
- **Real AI Reasoning**: Shows actual LLM decision-making process
- **Domain-Specific Analysis**: Adapts analysis based on research field
- **Quality Assessment**: Comprehensive scoring with confidence metrics
- **Actionable Recommendations**: AI-generated improvement suggestions
- **Cost Tracking**: Monitor API usage and costs

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   SENSE Phase   │    │   PLAN Phase     │    │   ACT Phase     │
│                 │    │                  │    │                 │
│ • LLM analyzes  │───▶│ • LLM creates    │───▶│ • LLM executes  │
│   paper content │    │   analysis       │    │   comprehensive │
│ • Extracts      │    │   strategy       │    │   analysis      │
│   insights      │    │ • Tailors to     │    │ • Generates     │
│ • Identifies    │    │   paper type     │    │   insights      │
│   domain        │    │ • Plans tasks    │    │ • Provides      │
│                 │    │                  │    │   reasoning     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📦 Installation

### 1. Clone and Setup
```bash
cd ~/agentic_research_analyzer_ai
pip install -r requirements.txt
```

### 2. Configure API Keys
```bash
# Option 1: Environment variables (recommended)
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"

# Option 2: Configuration file
cp config_default.yaml config.yaml
# Edit config.yaml with your API keys
```

### 3. Test Installation
```bash
python -m src.cli --help
```

## 🎯 Usage

### Basic Analysis
```bash
# Analyze a paper with AI
python -m src.cli --mode ai --paper sample_data/paper1.txt

# Use specific provider
python -m src.cli --mode ai --paper sample_data/paper1.txt --provider openai

# Get summary output
python -m src.cli --mode ai --paper sample_data/paper1.txt --output-format summary
```

### Advanced Usage
```bash
# JSON output with detailed reasoning
python -m src.cli --mode ai --paper sample_data/paper1.txt --output-format json --output results.json

# Compare with rule-based analysis
python -m src.cli --mode rule --paper sample_data/paper1.txt --output-format summary
python -m src.cli --mode ai --paper sample_data/paper1.txt --output-format summary

# Process from stdin
cat paper.txt | python -m src.cli --mode ai --stdin --output-format detailed
```

### Configuration Options
```bash
# Use custom config file
python -m src.cli --mode ai --paper sample_data/paper1.txt --config config.yaml

# Enable debug mode
python -m src.cli --mode ai --paper sample_data/paper1.txt --debug

# Show LLM reasoning
python -m src.cli --mode ai --paper sample_data/paper1.txt --show-reasoning
```

## 🔧 Configuration

### Provider Configuration
```yaml
# config.yaml
openai:
  api_key: ${OPENAI_API_KEY}
  default_model: "gpt-4"
  max_tokens: 2000

anthropic:
  api_key: ${ANTHROPIC_API_KEY}
  default_model: "claude-3-sonnet-20240229"
  max_tokens: 2000

local:
  endpoint: "http://localhost:11434/api/generate"
  default_model: "llama2"
```

### Analysis Configuration
```yaml
analysis:
  provider_priority: ["openai", "anthropic", "local"]
  sense_temperature: 0.3
  plan_temperature: 0.4
  act_temperature: 0.3
  enable_fallback: true
```

## 📊 Output Formats

### Detailed Output (Default)
- Complete metadata table
- Task execution results
- Overall assessment with reasoning
- Detailed findings per task
- All recommendations with priorities

### Summary Output
- Key paper characteristics
- Quality score and assessment
- Top strengths and weaknesses
- Priority recommendations

### JSON Output
- Machine-readable format
- Complete analysis results
- LLM reasoning and metadata
- Token usage and cost information

## 🧠 AI Reasoning Transparency

The AI-powered analyzer provides full transparency into LLM reasoning:

```json
{
  "llm_reasoning": {
    "sense_phase": "I analyzed the paper structure and identified...",
    "plan_phase": "Based on the paper type, I created a strategy focusing on...",
    "act_phase": "My analysis revealed several key findings...",
    "overall_assessment": "Synthesizing all analyses, I conclude...",
    "recommendations": "The most impactful improvements would be..."
  }
}
```

## 🔄 Comparison with Rule-Based Version

| Feature | Rule-Based | AI-Powered |
|---------|------------|------------|
| Analysis Depth | Pattern matching | Deep understanding |
| Adaptability | Fixed rules | Context-aware |
| Domain Knowledge | Limited | Extensive |
| Reasoning | Deterministic | Intelligent |
| Insights | Surface-level | Comprehensive |
| Recommendations | Generic | Tailored |

### Running Comparisons
```bash
# Rule-based analysis
python -m src.cli --mode rule --paper sample_data/paper1.txt > rule_results.txt

# AI-powered analysis  
python -m src.cli --mode ai --paper sample_data/paper1.txt > ai_results.txt

# Compare results
diff rule_results.txt ai_results.txt
```

## 🛠️ Error Handling & Fallbacks

### Robust Error Handling
- **API Failures**: Automatic retry with exponential backoff
- **Rate Limiting**: Built-in rate limiting and queue management
- **Provider Fallback**: Automatic switching between providers
- **Graceful Degradation**: Falls back to rule-based analysis if needed

### Cost Management
- **Usage Tracking**: Monitor token usage and estimated costs
- **Budget Limits**: Set maximum cost per analysis
- **Cost Warnings**: Alerts when approaching budget limits

## 📈 Performance & Scalability

### Optimization Features
- **Token Optimization**: Efficient prompt engineering
- **Caching**: Cache LLM responses for repeated analyses
- **Batch Processing**: Process multiple papers efficiently
- **Async Support**: Non-blocking API calls

### Monitoring
```bash
# View analysis logs
tail -f analysis_logs.txt

# Check token usage
python -m src.cli --mode ai --paper sample_data/paper1.txt --show-stats
```

## 🔍 Sample Analysis Results

### Input Paper Characteristics
- **Type**: Machine Learning Research Paper
- **Domain**: Computer Vision
- **Length**: 8,500 words
- **Complexity**: 7.5/10

### AI Analysis Output
```
╭─ Analysis Summary ─╮
│ Paper Type: research │
│ Domain: computer vision │
│ Quality Score: 8.2/10 │
│ Complexity: 7.5/10 │
╰────────────────────╯

✓ Strengths:
  • Novel architecture with strong theoretical foundation
  • Comprehensive experimental evaluation
  • Clear presentation and reproducible results

⚠ Areas for Improvement:
  • Limited comparison with recent baselines
  • Ablation studies could be more thorough
  • Discussion of limitations needs expansion

🔍 Top Recommendations:
  1. Add comparison with 2024 state-of-the-art methods
  2. Include failure case analysis and discussion
  3. Provide more detailed computational complexity analysis
```

## 🚀 Demo Script

For faculty demonstrations:

```bash
# Quick demo with sample papers
./run_demo.sh

# Compare rule-based vs AI analysis
./run_comparison_demo.sh

# Show different output formats
./run_format_demo.sh
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built upon the foundation of the rule-based research analyzer
- Inspired by advances in LLM reasoning capabilities
- Designed for CS faculty demonstrations of AI reasoning systems

---

**Note**: This AI-powered version demonstrates real artificial intelligence reasoning, making it significantly more impressive for academic demonstrations compared to rule-based pattern matching approaches.
