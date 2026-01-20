# AgingResearchAI

Multi-model AI system for aging and metabolic disease drug discovery.

## Features

- **3-LLM Debate System**: Gemini (propose) + DeepSeek (critique) + Kimi (judge)
- **RAG Knowledge Base**: PubMed literature with semantic search
- **ADMET Prediction**: Drug-likeness and toxicity via DeepChem/RDKit
- **LangGraph Orchestration**: Stateful research pipelines

## Target Diseases

- Type 2 Diabetes (T2D)
- NAFLD/NASH
- Sarcopenia

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Configure API keys
cp .env.example .env

# Test ADMET (no API keys needed)
python scripts/skill_runner.py admet "CCO" --quick

# Search literature
python scripts/skill_runner.py search "AMPK diabetes" --load

# Run debate
python scripts/skill_runner.py debate "Metformin mechanism" --claim "Works via AMPK"

# Start API server
python scripts/skill_runner.py server --port 8000
```

## Cost Profile

| Model | Role | Cost |
|-------|------|------|
| Gemini | Proposer | FREE |
| DeepSeek | Critic | ~$0.14/M tokens |
| Kimi | Judge | ~$0.20/M tokens |
| **Daily** | | **~$1-5** |

## Documentation

See [CLAUDE.md](CLAUDE.md) for full documentation.

## License

MIT
