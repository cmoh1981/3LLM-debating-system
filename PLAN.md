# AgingResearchAI: Multi-Model System for Drug Discovery

## Overview

An AI-powered research system for aging and metabolic disease drug discovery, combining:
- **Gemini 2.5 Flash** (80% of tasks - free tier)
- **Claude Sonnet** (20% of tasks - complex reasoning)
- **Lobster AI** (bioinformatics: RNA-seq, dataset discovery, literature mining)
- **LangGraph** (stateful multi-agent orchestration with persistence)
- **DeepChem** (deep learning for ADMET, toxicity, molecular properties)

### Core Capabilities
1. **Pathogenesis Discovery** - Identify disease mechanisms
2. **Drug Target Identification** - Find and prioritize targets
3. **In Silico ADMET Validation** - Safety and efficacy prediction
4. **Experiment Suggestion** - Validation study design
5. **Patent Landscape Analysis** - Freedom to operate
6. **Real-time PubMed Monitoring** - Stay current

### Target Diseases (Phase 1)
- Type 2 Diabetes (T2D)
- NAFLD/NASH
- Sarcopenia

---

## Key Design Principles

Based on **LongevityBench** findings and expert reviews:

| Principle | Implementation |
|-----------|----------------|
| LLMs write code, don't analyze raw data | Gemini generates scripts, Python/Lobster executes |
| Classification > Regression | Binary/categorical outputs only |
| Tool interpretation > Direct prediction | LLM explains what tools found |
| Evidence-linked claims | Every statement needs citation |
| Structured prompts | XML-like templates for consistency |
| No uncited claims | Hard rule - every claim traces to source |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         User Interface / API                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                      LangChain Orchestrator                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ Query Router │  │ Chain Manager│  │ Memory Store │  │ Scheduler  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│  Gemini 2.5 Flash │    │   Claude Sonnet   │    │    Lobster AI     │
│                   │    │                   │    │                   │
│  • 80% of tasks   │    │  • 20% of tasks   │    │  • Bioinformatics │
│  • RAG retrieval  │    │  • Final synthesis│    │  • RNA-seq        │
│  • Code generation│    │  • Hypothesis gen │    │  • Dataset search │
│  • Data extraction│    │  • Risk assessment│    │  • Literature     │
│  • Patent search  │    │  • Experiment plan│    │  • QC & analysis  │
│                   │    │  • Report writing │    │                   │
│  FREE TIER        │    │  PAID (selective) │    │  SPECIALIZED      │
└───────────────────┘    └───────────────────┘    └───────────────────┘
          │                         │                         │
          └─────────────────────────┼─────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Tool Execution Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ Python       │  │ RDKit/       │  │ ADMET APIs   │  │ Database   │  │
│  │ Sandbox      │  │ DeepChem     │  │ (pkCSM etc)  │  │ Queries    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────┐
│                         RAG Knowledge Base                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │ PubMed  │ │ Patents │ │ DrugBank│ │  KEGG   │ │  ADMET  │          │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐          │
│  │ UniProt │ │OpenGenes│ │  GTEx   │ │  TCGA   │ │ ChEMBL  │          │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Global JSON Schema (All Modules)

Every module output must conform to this schema:

```json
{
  "run_id": "uuid",
  "module": "string",
  "model_used": "gemini | claude | lobster",
  "status": "ok | needs_review | failed",
  "timestamp": "ISO8601",

  "summary": "string",

  "claims": [
    {
      "text": "string",
      "confidence": 0.0-1.0,
      "evidence_tier": "tier1 | tier2 | tier3",
      "evidence": [
        {"type": "literature", "pmid": "string", "quote": "string"},
        {"type": "database", "source": "string", "id": "string"},
        {"type": "computed", "tool": "string", "artifact_id": "string"}
      ]
    }
  ],

  "artifacts": [
    {
      "id": "string",
      "type": "table | figure | code | file",
      "path": "string",
      "provenance": {
        "code_version": "git_sha",
        "inputs": ["artifact_id", "..."],
        "params": {}
      }
    }
  ],

  "next_actions": [
    {"action": "string", "priority": "P0 | P1 | P2", "reason": "string"}
  ],

  "warnings": ["string"],
  "errors": [{"code": "string", "message": "string"}]
}
```

### Evidence Tiers

| Tier | Criteria | Confidence |
|------|----------|------------|
| **Tier 1** | Replicated + Causal support + Literature | 🟢 High |
| **Tier 2** | Replicated association | 🟡 Medium |
| **Tier 3** | Single analysis only | 🔴 Low - needs validation |

---

## Model Routing Rules

```python
ROUTING_RULES = {
    # Gemini (Free, 80% of tasks)
    "literature_search": "gemini",
    "dataset_discovery": "gemini",
    "code_generation": "gemini",
    "data_extraction": "gemini",
    "patent_search": "gemini",
    "initial_screening": "gemini",
    "api_orchestration": "gemini",

    # Claude (Paid, 20% of tasks - critical reasoning)
    "pathogenesis_synthesis": "claude",
    "target_prioritization": "claude",
    "risk_assessment": "claude",
    "experiment_design": "claude",
    "final_interpretation": "claude",
    "report_generation": "claude",
    "evidence_adjudication": "claude",

    # Lobster AI (Specialized bioinformatics)
    "rnaseq_analysis": "lobster",
    "scrna_analysis": "lobster",
    "geo_dataset_search": "lobster",
    "differential_expression": "lobster",
    "pathway_enrichment": "lobster",
    "quality_control": "lobster",
}
```

---

## Implementation Phases

### Phase 1: Foundation (Week 1-2)

#### 1.1 Project Setup
- [ ] Initialize Python project with Poetry
- [ ] Set up LangChain with Gemini + Claude
- [ ] Install and configure Lobster AI
- [ ] Set up ChromaDB for RAG
- [ ] Create JSON schema validators
- [ ] Set up code execution sandbox

#### 1.2 Core Infrastructure
```
deepagents/
├── src/
│   ├── core/
│   │   ├── schema.py           # JSON schema definitions
│   │   ├── router.py           # Model routing logic
│   │   ├── evidence.py         # Evidence tier classification
│   │   └── sandbox.py          # Code execution sandbox
│   ├── models/
│   │   ├── gemini_client.py
│   │   ├── claude_client.py
│   │   └── lobster_client.py
│   ├── rag/
│   │   ├── knowledge_base.py
│   │   ├── embeddings.py
│   │   └── retriever.py
│   └── utils/
│       └── validators.py
├── config/
│   ├── settings.yaml
│   └── prompts/                # XML prompt templates
├── data/
└── tests/
```

---

### Phase 2: Research Modules (Week 3-4)

#### 2.1 Pathogenesis Discovery Module

**Workflow:**
```
Disease Input
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Lobster AI: Dataset Discovery                              │
│  - Search GEO/SRA for relevant datasets                     │
│  - Identify bulk RNA-seq, scRNA-seq studies                 │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Lobster AI: Analysis                                       │
│  - QC and normalization                                     │
│  - Differential expression                                  │
│  - Pathway enrichment (ssGSEA)                              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Gemini: Literature RAG                                     │
│  - Retrieve relevant papers                                 │
│  - Extract mechanism evidence                               │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Claude: Synthesis                                          │
│  - Integrate omics + literature                             │
│  - Propose pathogenic mechanisms                            │
│  - Assign evidence tiers                                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Ranked Pathogenic Pathways (with evidence chains)
```

#### 2.2 Target Identification Module

**Workflow:**
```
Pathogenic Pathway
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Gemini: Code Generation                                    │
│  - Write target scoring scripts                             │
│  - Query protein databases                                  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Tool Execution: Target Analysis                            │
│  - Druggability scoring                                     │
│  - Tissue specificity (GTEx)                                │
│  - Off-target prediction                                    │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Claude: Prioritization                                     │
│  - Evaluate druggability                                    │
│  - Assess safety risks                                      │
│  - Rank targets                                             │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
Prioritized Target List (with rationale)
```

#### 2.3 ADMET Module

**Workflow:**
```
Candidate Compound (SMILES)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Gemini: Generate ADMET Script                              │
│  - RDKit property calculations                              │
│  - API calls to pkCSM, SwissADME                            │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Tool Execution: ADMET Prediction                           │
│  - Physicochemical properties                               │
│  - ADMET endpoint predictions                               │
│  - Toxicity alerts                                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Claude: Interpretation (Classification ONLY)               │
│  - Risk classification: High/Medium/Low                     │
│  - Liability identification                                 │
│  - Modification suggestions                                 │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
ADMET Report (classification, NOT regression)
```

---

### Phase 3: Additional Features (Week 5-6)

#### 3.1 Experiment Suggester
- Generate validation experiments for hypotheses
- Prioritize by information gain per cost
- Include controls, readouts, expected results
- Estimate cost tiers ($, $$, $$$)

#### 3.2 Patent Landscape Analyzer
- Search Google Patents, Lens.org
- Assess freedom to operate
- Identify white space opportunities
- Track competitor activity

#### 3.3 Real-time PubMed Monitor
- Keyword and author watchlists
- Daily/weekly alert digests
- Auto-relevance scoring via Gemini
- Critical paper summarization

---

## Prompt Templates

### Template 1: Pathogenesis (Claude)

```xml
<task>
Analyze patient/experimental data to identify disease mechanisms.
</task>

<question>{question}</question>

<options>{options}</options>

<patient_data>
    <demographic>{demographic}</demographic>
    <clinical_history>{clinical_history}</clinical_history>
    <biomarkers>{biomarkers}</biomarkers>
    <medications>{medications}</medications>
</patient_data>

<omics_results>
    <differential_expression>{de_results}</differential_expression>
    <pathway_enrichment>{pathway_results}</pathway_enrichment>
</omics_results>

<literature_context>{rag_literature}</literature_context>

<instructions>
1. Analyze each data category independently
2. Identify risk factors and protective factors
3. Propose pathogenic mechanism hypothesis
4. Assign evidence tier (1/2/3) to each claim
5. Output classification (not regression)
6. Cite all evidence sources
</instructions>

<output_format>
JSON conforming to global schema
</output_format>
```

### Template 2: Target Discovery (Gemini → Code)

```xml
<task>
Generate Python code to analyze omics data for target identification.
</task>

<research_question>{question}</research_question>

<data_context>
    <organism>{organism}</organism>
    <tissue>{tissue}</tissue>
    <data_type>{omics_type}</data_type>
    <data_path>{data_path}</data_path>
</data_context>

<analysis_requirements>
    <primary_analysis>{analysis_type}</primary_analysis>
    <tools_available>
    pandas, numpy, scipy, scanpy, gseapy, pydeseq2, statsmodels
    </tools_available>
</analysis_requirements>

<instructions>
1. Write complete, executable Python code
2. Include QC, analysis, and visualization
3. Output structured results (JSON)
4. Do NOT hallucinate - use only provided data
5. Add comments explaining each step
</instructions>
```

### Template 3: ADMET (Claude Interpretation)

```xml
<task>
Interpret ADMET predictions and classify compound risk.
</task>

<compound>
    <id>{compound_id}</id>
    <smiles>{smiles}</smiles>
    <target>{target}</target>
</compound>

<computed_properties>
    <physicochemical>
    MW: {mw}, LogP: {logp}, HBD: {hbd}, HBA: {hba}, TPSA: {tpsa}
    </physicochemical>

    <admet_predictions>
    Absorption: {absorption}
    Distribution: {distribution}
    Metabolism: {metabolism}
    Excretion: {excretion}
    Toxicity: {toxicity}
    </admet_predictions>
</computed_properties>

<reference_compounds>{similar_drugs}</reference_compounds>

<instructions>
1. Interpret values (do NOT predict new values)
2. Compare to reference compounds
3. CLASSIFY risk level (High/Medium/Low) - no regression
4. Identify specific liabilities
5. Suggest modifications if needed
</instructions>
```

### Template 4: Experiment Suggestion (Claude)

```xml
<task>
Design validation experiments for the hypothesis.
</task>

<hypothesis>{hypothesis}</hypothesis>

<target_context>
    <target>{target}</target>
    <disease>{disease}</disease>
    <mechanism>{mechanism}</mechanism>
    <evidence_tier>{tier}</evidence_tier>
</target_context>

<constraints>
    <budget>{budget_tier}</budget>
    <timeline>{timeline}</timeline>
    <available_models>{models}</available_models>
</constraints>

<instructions>
1. Suggest experiments: simplest to most complex
2. Include positive and negative controls
3. Specify readouts and expected results
4. Estimate cost tier ($/$$/$$)
5. Prioritize by information gain per dollar
6. Include "kill experiments" (fastest disproof)
</instructions>
```

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| **Agent Orchestration** | LangGraph (stateful, persistent agents) |
| **LLM Framework** | LangChain (model abstraction, tools) |
| Vector DB | ChromaDB (local) |
| Embeddings | PubMedBERT / sentence-transformers |
| LLM - Primary | Google Gemini 2.5 Flash (free) |
| LLM - Reasoning | Anthropic Claude Sonnet |
| Bioinformatics | Lobster AI |
| **ADMET/Molecular ML** | DeepChem (GCN, AttentiveFP, Tox21) |
| Molecular | RDKit, DataMol |
| Code Sandbox | RestrictedPython / Docker |
| Web Framework | FastAPI |
| Monitoring | Scheduled tasks (cron) |

---

## LangGraph Agent Architecture

LangGraph provides stateful, persistent multi-agent orchestration with:
- **Graph-based workflows**: Nodes = operations, Edges = flow control
- **Persistent state**: Checkpointing for long-running analyses
- **Human-in-the-loop**: Intervention points for expert review
- **Multi-agent coordination**: Specialized agents working together

### Agent Graph Structure

```
                    ┌─────────────────┐
                    │   User Input    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Router Agent   │ (Gemini - decides workflow)
                    │  (StateGraph)   │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Pathogenesis    │ │ Target          │ │ ADMET           │
│ Agent           │ │ Agent           │ │ Agent           │
│ (Lobster+Claude)│ │ (Gemini+Claude) │ │ (DeepChem+Claude│
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                    ┌────────▼────────┐
                    │ Synthesis Agent │ (Claude - final integration)
                    │ (with memory)   │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Human Review    │ (checkpoint for approval)
                    │ (optional)      │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Final Report   │
                    └─────────────────┘
```

### LangGraph State Schema

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph
from operator import add

class ResearchState(TypedDict):
    # Input
    disease: str
    tissue: str | None

    # Accumulated results
    datasets: list[str]
    omics_results: Annotated[list[dict], add]
    literature: Annotated[list[dict], add]
    targets: list[dict]
    admet_results: list[dict]

    # Evidence tracking
    claims: Annotated[list[dict], add]
    evidence_tier: str

    # Control flow
    current_step: str
    needs_human_review: bool
    errors: list[str]

    # Memory
    messages: Annotated[list, add]
```

---

## DeepChem Integration

DeepChem provides deep learning models for molecular property prediction:

### Supported Models

| Model | Use Case | Input |
|-------|----------|-------|
| **AttentiveFPModel** | ADMET prediction | SMILES → Graph |
| **GCNModel** | Toxicity (Tox21) | Molecular graph |
| **ChemBERTa** | Property prediction | SMILES tokens |
| **DMPNNModel** | Binding affinity | Directed message passing |
| **GroverModel** | Pre-trained embeddings | Self-supervised |

### Available Datasets for Training/Validation

| Dataset | Task | Relevance |
|---------|------|-----------|
| **Tox21** | 12 toxicity endpoints | Safety screening |
| **SIDER** | Side effects | Adverse reactions |
| **BBBP** | Blood-brain barrier | CNS drug design |
| **ClinTox** | Clinical toxicity | Clinical translation |
| **Clearance** | Drug clearance | Metabolism |
| **ChEMBL** | Bioactivity | Target engagement |

### DeepChem ADMET Pipeline

```python
import deepchem as dc
from deepchem.models import AttentiveFPModel

# Featurizer for graph-based models
featurizer = dc.feat.MolGraphConvFeaturizer()

# Load pre-trained toxicity model
tox_model = AttentiveFPModel(
    n_tasks=12,  # Tox21 has 12 endpoints
    mode='classification',
    learning_rate=0.001
)
tox_model.restore()  # Load pre-trained weights

# Predict
smiles = ["CCO", "c1ccccc1"]
features = featurizer.featurize(smiles)
predictions = tox_model.predict(features)
```

### Integration with LLM Interpretation

Following LongevityBench principle: **DeepChem predicts, Claude interprets**

```
SMILES Input
    │
    ▼
┌─────────────────────────────────────┐
│ DeepChem Prediction (Tools)         │
│ - Tox21 toxicity scores             │
│ - ADMET property predictions        │
│ - Molecular fingerprints            │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Claude Interpretation (LLM)         │
│ - Classify risk: High/Medium/Low    │
│ - Compare to reference drugs        │
│ - Suggest modifications             │
│ - Generate human-readable report    │
└─────────────────────────────────────┘
    │
    ▼
ADMET Report with Evidence
```

---

## Janggu Integration (Genomics Deep Learning)

Janggu provides deep learning infrastructure for genomic sequence analysis:
- GitHub: https://github.com/BIMSBbioinfo/janggu

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **Bioseq** | DNA sequence loading from FASTA, encoding (one-hot, dinucleotide) |
| **Cover** | Coverage data from BAM, BigWig, BED files |
| **GenomicIndexer** | Efficient genomic region iteration |
| **Keras Integration** | Seamless deep learning model training |
| **BigWig Export** | Convert predictions to genome browser tracks |
| **Variant Effects** | Predict impact of genetic variants |

### Supported File Formats

```
FASTA  → DNA sequences (reference genome)
BAM    → Aligned reads (RNA-seq, ChIP-seq, ATAC-seq)
BigWig → Coverage tracks (signal data)
BED    → Genomic intervals (peaks, regions of interest)
GFF    → Gene annotations
```

### Applications for Aging Research

1. **Epigenetic Aging Clocks**
   - Train models on methylation data
   - Predict biological age from epigenetic marks

2. **Regulatory Element Prediction**
   - TF binding site prediction
   - Enhancer/promoter identification
   - Chromatin accessibility (ATAC-seq)

3. **Variant Effect Prediction**
   - Assess impact of aging-associated SNPs
   - Predict functional consequences of mutations

4. **Gene Expression Prediction**
   - Predict expression from sequence
   - Identify regulatory variants

### Janggu Pipeline Example

```python
from janggu.data import Bioseq, Cover, GenomicIndexer

# Define regions of interest
roi = GenomicIndexer.create_from_file(
    'regions.bed',
    binsize=200,
    stepsize=50
)

# Load DNA sequences
dna = Bioseq.create_from_refgenome(
    name='dna',
    refgenome='hg38.fa',
    roi=roi,
    order=2  # dinucleotide encoding
)

# Load ATAC-seq signal as labels
labels = Cover.create_from_bigwig(
    name='atac',
    bigwigfiles=['atac_seq.bw'],
    roi=roi,
    resolution=50
)

# Train Keras model
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten

model = Sequential([
    Conv1D(32, 11, activation='relu'),
    Flatten(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(dna, labels, epochs=10)

# Export predictions as BigWig
from janggu import export_bigwig
export_bigwig(model, dna, 'predictions.bw')
```

### Integration with AgingResearchAI

```
Genomic Data (BAM, BED, BigWig)
    │
    ▼
┌─────────────────────────────────────┐
│ Janggu Data Loading                 │
│ - Bioseq: DNA sequence encoding     │
│ - Cover: Epigenetic signals         │
│ - GenomicIndexer: Region iteration  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Deep Learning Models (Keras)        │
│ - CNN for sequence motifs           │
│ - Attention for long-range          │
│ - Multi-task for multiple marks     │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Claude Interpretation               │
│ - Identify regulatory elements      │
│ - Link to aging pathways            │
│ - Prioritize variants               │
└─────────────────────────────────────┘
    │
    ▼
Genomic Insights with Evidence
```

---

## File Structure

```
deepagents/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── schema.py           # JSON schema & validators
│   │   ├── router.py           # Model routing
│   │   ├── evidence.py         # Evidence tier logic
│   │   └── sandbox.py          # Code execution
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gemini_client.py
│   │   ├── claude_client.py
│   │   └── lobster_client.py
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── pathogenesis_agent.py
│   │   ├── target_agent.py
│   │   ├── admet_agent.py
│   │   ├── experiment_agent.py
│   │   └── patent_agent.py
│   │
│   ├── chains/
│   │   ├── __init__.py
│   │   ├── discovery_chain.py
│   │   ├── validation_chain.py
│   │   └── synthesis_chain.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── knowledge_base.py
│   │   ├── embeddings.py
│   │   ├── retriever.py
│   │   └── patent_index.py
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── pubmed_watcher.py
│   │   ├── alert_manager.py
│   │   └── scheduler.py
│   │
│   ├── admet/
│   │   ├── __init__.py
│   │   ├── predictor.py
│   │   ├── interpreter.py
│   │   └── optimizer.py
│   │
│   └── api/
│       ├── __init__.py
│       ├── main.py
│       └── routes.py
│
├── config/
│   ├── settings.yaml
│   ├── watchlist.yaml          # PubMed monitoring keywords
│   └── prompts/
│       ├── pathogenesis.xml
│       ├── target_discovery.xml
│       ├── admet.xml
│       └── experiment.xml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
│
├── scripts/
│   ├── build_rag.py
│   ├── daily_monitor.py
│   └── run_pipeline.py
│
├── tests/
│   ├── test_schema.py
│   ├── test_routing.py
│   ├── test_agents.py
│   └── test_chains.py
│
├── notebooks/
│   └── exploration.ipynb
│
├── pyproject.toml
└── README.md
```

---

## Acceptance Tests

### Schema Compliance
- [ ] Every module returns valid JSON
- [ ] All required fields present
- [ ] Evidence array non-empty for all claims

### Evidence Rules
- [ ] Zero uncited claims in final output
- [ ] Every claim has evidence_tier assigned
- [ ] Computed results link to artifact IDs

### Reproducibility
- [ ] Same input → same output (deterministic)
- [ ] All artifacts have provenance recorded
- [ ] Code versions tracked

### Model Routing
- [ ] Correct model called for each task type
- [ ] Fallback handling works
- [ ] Cost stays within budget

### Integration
- [ ] End-to-end pipeline completes
- [ ] Lobster AI integrates correctly
- [ ] RAG retrieval returns relevant chunks

---

## Cost Estimate

| Component | Cost |
|-----------|------|
| Gemini 2.5 Flash | Free (1,500 req/day) |
| Claude Sonnet | ~$5-10/day heavy use |
| Lobster AI | Depends on compute |
| ChromaDB | Free (local) |
| **Total** | **~$5-15/day** |

---

## Next Steps

1. **Set up project structure** ← START HERE
2. **Configure Lobster AI** for bioinformatics
3. **Build JSON schema validators**
4. **Create model clients** (Gemini, Claude, Lobster)
5. **Implement RAG knowledge base**
6. **Build first chain** (Pathogenesis)
7. **Add remaining modules**
8. **Create API endpoints**
9. **Set up monitoring**
10. **Validate on test cases**
