# AI Underwriting Demo (Russell 2000) — Multi‑Agent + RAG 
*A Streamlit application that simulates a small‑business underwriting workflow using LLM Agents + Retrieval‑Augmented Generation (RAG).*

---
## Overview

This project demonstrates how modern AI systems can accelerate the underwriting process for small‑business loan applications. It combines:

- **LLM‑powered multi‑agent pipeline**
- **Rule‑based financial risk scoring**
- **RAG retrieval** from a small industry + policy knowledge base
- **Streamlit UI** for interactive exploration

The system evaluates a business using signals such as revenue, debt, credit score, Google review metrics, and industry risk. It then retrieves relevant policy & industry knowledge using embeddings, and generates a final underwriting memo and decision.

---
## Features

### Multi‑Agent Pipeline
1. **Collector Agent**  
   Extracts signals & engineered features (review strength, debt ratio, industry risk).

2. **Risk Analyst Agent**  
   Computes a normalized risk score (0–100) using weighted financial & operational factors.

3. **RAG Agent**  
   Retrieves industry‑specific trends and bank policy documents from an embedded knowledge base.

4. **Underwriter Agent (LLM)**  
   Produces a structured memo with:
   - Summary  
   - Strengths & risks  
   - Decision (APPROVE / DECLINE / REVIEW)  
   - Confidence score  

---
## Architecture
```scss

             ┌────────────────────┐
             │  Collector Agent   │
             │  (Feature Engine)  │
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Risk Analyst       │
             │ (Scoring Model)    │
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │    RAG Agent       │
             │ (Policy + Industry)│
             └─────────┬──────────┘
                       │
                       ▼
             ┌────────────────────┐
             │ Underwriter (LLM)  │
             │ Final Decision     │
             └────────────────────┘

```

---
## Project Structure

```yaml
.
├── app.py                              # Streamlit UI
├── underwriting_agents_rag.py          # Multi-agent + RAG logic
├── generate_russell2000_businesses.py  # Russell 2000 business generator
├── requirements.txt                    # Python dependencies
├── .gitignore
├── README.md
└── RUSSELL2000_GENERATOR.md            # Business generator documentation
```

### Business Data Generator
The project now includes a sophisticated business data generator based on actual Russell 2000 index companies. See [RUSSELL2000_GENERATOR.md](RUSSELL2000_GENERATOR.md) for detailed documentation.

**Key Features:**
- 100 real Russell 2000 company names
- 10 industry categories with realistic financial profiles
- Standalone CLI script or importable Python module
- Generates CSV files with company data

**Usage:**
```bash
# Generate 50 businesses
python generate_russell2000_businesses.py 50

# Or use as a module
from generate_russell2000_businesses import generate_russell2000_businesses
df = generate_russell2000_businesses(n=50)
```

---
## RAG Knowledge Base
The project includes a small embedded RAG dataset containing:
- Industry risk insights
- Bank underwriting policies
- Customer‑review signal guidelines
- Embeddings use text-embedding-3-small.

---
## How the Risk Score Works
Risk is determined using weighted factors:
Credit score
Review strength
Revenue
Debt ratio
Industry baseline risk
Mapped to a 0–100 score (higher = safer).

---
## Underwriter Agent Output
Example structure:

**SUMMARY**: \<business overview\>

**STRENGTHS**:
- \<item\>
  
**RISKS**:
- \<item\>

**DECISION**: APPROVE | DECLINE | REVIEW

**CONFIDENCE**: 0.78


---
## Contributing
PRs and enhancements welcome! Ideas include:
- Expanded RAG datasets
- More agents
- Real review scraping
- Machine‑learned risk models

---
## License
MIT License — free to use and modify.

---
