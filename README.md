# AI Underwriting Demo — Multi‑Agent + RAG  
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
├── app.py # Streamlit UI
├── underwriting_agents_rag.py # Multi-agent + RAG logic
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md
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
SUMMARY: \<business overview\>

STRENGTHS:
- \<item\>
  
RISKS:
- \<item\>

DECISION: APPROVE | DECLINE | REVIEW
CONFIDENCE: 0.78


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
