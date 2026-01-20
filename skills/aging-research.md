# /aging-research - AgingResearchAI Skill

Multi-LLM system for aging and metabolic disease drug discovery.

## When to Use

Use this skill when the user asks about:
- **Aging biology research** (T2D, NAFLD/NASH, Sarcopenia)
- **Literature search** with PubMed citations
- **Scientific claim verification** via multi-LLM debate
- **ADMET prediction** for drug candidates
- **Drug target discovery** for metabolic diseases

## Commands

### 1. Literature Search
```bash
python scripts/skill_runner.py search "<query>" --load --top-k 10
```

Example:
```bash
python scripts/skill_runner.py search "AMPK insulin resistance T2D" --load
```

### 2. Multi-LLM Debate (Claim Verification)
```bash
python scripts/skill_runner.py debate "<topic>" --claim "<claim to verify>"
```

Example:
```bash
python scripts/skill_runner.py debate "Metformin mechanism of action" --claim "Metformin primarily works through AMPK activation"
```

### 3. ADMET Prediction
```bash
# Quick check (RDKit only)
python scripts/skill_runner.py admet "<SMILES>" --quick

# Full prediction (DeepChem)
python scripts/skill_runner.py admet "<SMILES>"
```

Example:
```bash
python scripts/skill_runner.py admet "CC(=O)Nc1ccc(O)cc1" --quick  # Acetaminophen
```

### 4. Research Pipeline
```bash
python scripts/skill_runner.py pipeline <disease> --tissue <tissue> --workflow <type>
```

Example:
```bash
python scripts/skill_runner.py pipeline T2D --tissue liver --workflow pathogenesis
```

### 5. Start API Server
```bash
python scripts/skill_runner.py server --port 8000
```

### 6. Check Costs
```bash
python scripts/skill_runner.py costs
```

## Python API (Alternative)

```python
# Literature Search
from src.rag import KnowledgeBase, PubMedLoader, HybridRetriever

kb = KnowledgeBase(persist_directory="data/embeddings")
loader = PubMedLoader(query="AMPK diabetes", max_results=50)
kb.add_documents(loader.load(), collection="literature")

retriever = HybridRetriever(kb, collection="literature")
results = retriever.retrieve("insulin resistance mechanism", top_k=5)

# Multi-LLM Debate
from src.chains import DebateEngine
from src.models import GeminiClient, DeepSeekClient, KimiClient

engine = DebateEngine(clients={
    "gemini": GeminiClient(),
    "deepseek": DeepSeekClient(),
    "kimi": KimiClient(),
})
result = engine.debate(topic="AMPK in T2D", initial_claims=[...])

# ADMET
from src.admet import quick_admet_check
result = quick_admet_check("CCO")  # Returns dict with properties

# Pipeline
from src.chains import ResearchOrchestrator
orchestrator = ResearchOrchestrator(enable_debate=True, use_rag=True)
result = orchestrator.run(disease="T2D", tissue="liver", workflow_type="pathogenesis")
```

## Target Diseases

| Disease | Aliases | Key Tissues |
|---------|---------|-------------|
| Type 2 Diabetes | T2D, T2DM | liver, muscle, pancreas, adipose |
| NAFLD/NASH | Fatty liver | liver, adipose |
| Sarcopenia | Muscle wasting | muscle |

## Cost Profile

| Model | Role | Cost |
|-------|------|------|
| Gemini 2.5 Flash | Proposer | **FREE** |
| Grok (xAI) | Critic | ~$5/M input, $15/M output |
| DeepSeek V3.2 | Judge | ~$0.14/M tokens |

## Evidence Requirements

All claims MUST cite evidence:
- `PMID:12345678` - PubMed citation
- `UniProt:P12345` - Database ID
- `Computed:artifact_id` - ML prediction artifact

## Prerequisites

```bash
# Install
pip install -e .

# Configure API keys in .env
GOOGLE_API_KEY=xxx      # Required (free from aistudio.google.com)
XAI_API_KEY=xxx         # Required (from console.x.ai)
DEEPSEEK_API_KEY=xxx    # Required (from platform.deepseek.com)
NCBI_API_KEY=xxx        # Optional (for PubMed)
```
