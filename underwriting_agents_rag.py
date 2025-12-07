import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()  # reads OPENAI_API_KEY from env


# =====================================================
# 0. DATA STRUCTURES
# =====================================================

@dataclass
class Business:
    name: str
    industry: str
    revenue: float
    debt: float
    credit_score: int
    google_rating: float
    review_count: int


@dataclass
class CollectorOutput:
    business: Business
    review_strength: float
    debt_ratio: float
    industry_risk: float
    notes: str  # natural language description


@dataclass
class RiskOutput:
    business: Business
    risk_score: float
    risk_factors: Dict[str, Any]


@dataclass
class RAGContext:
    retrieved_chunks: List[str]


@dataclass
class UnderwriterOutput:
    memo: str
    decision: str
    confidence: float


# =====================================================
# 1. SYNTHETIC BUSINESS GENERATION
# =====================================================

def generate_sample_businesses(n: int = 20) -> pd.DataFrame:
    industries = [
        "Food & Beverage",
        "Retail",
        "Healthcare",
        "Construction",
        "Technology",
        "Personal Services",
    ]

    data = []
    rng = np.random.default_rng(42)

    for i in range(n):
        name = f"Business_{i+1}"
        industry = rng.choice(industries)
        revenue = rng.integers(80_000, 2_000_000)
        debt = rng.integers(10_000, 800_000)
        credit_score = rng.integers(580, 800)
        rating = np.round(rng.uniform(2.5, 5.0), 1)
        review_count = rng.integers(5, 1500)

        data.append(
            {
                "name": name,
                "industry": industry,
                "revenue": float(revenue),
                "debt": float(debt),
                "credit_score": int(credit_score),
                "google_rating": float(rating),
                "review_count": int(review_count),
            }
        )

    return pd.DataFrame(data)


# =====================================================
# 2. COLLECTOR AGENT
#    (feature engineering + short description)
# =====================================================

def _industry_risk_score(industry: str) -> float:
    table = {
        "Healthcare": 0.2,
        "Technology": 0.3,
        "Retail": 0.5,
        "Food & Beverage": 0.6,
        "Personal Services": 0.6,
        "Construction": 0.7,
    }
    return table.get(industry, 0.5)


def collector_agent(row: pd.Series) -> CollectorOutput:
    biz = Business(
        name=row["name"],
        industry=row["industry"],
        revenue=row["revenue"],
        debt=row["debt"],
        credit_score=row["credit_score"],
        google_rating=row["google_rating"],
        review_count=row["review_count"],
    )

    review_strength = biz.google_rating * np.log1p(biz.review_count)
    debt_ratio = biz.debt / biz.revenue if biz.revenue > 0 else 0.0
    industry_risk = _industry_risk_score(biz.industry)

    notes = (
        f"{biz.name} is a {biz.industry} business with revenue of "
        f"${biz.revenue:,.0f} and debt of ${biz.debt:,.0f}. It has a "
        f"credit score of {biz.credit_score}, a Google rating of "
        f"{biz.google_rating} based on {biz.review_count} reviews."
    )

    return CollectorOutput(
        business=biz,
        review_strength=float(review_strength),
        debt_ratio=float(debt_ratio),
        industry_risk=float(industry_risk),
        notes=notes,
    )


# =====================================================
# 3. RISK ANALYST AGENT
#    (rule-based scoring + factors)
# =====================================================

def risk_analyst_agent(collected: CollectorOutput, df_all: pd.DataFrame) -> RiskOutput:
    """
    Compute a relative risk_score 0–100 based on:
    - credit score
    - review strength
    - revenue
    - debt ratio
    - industry risk
    """
    biz = collected.business

    # Build comparison arrays from the whole dataset
    credit_scores = df_all["credit_score"].values
    revenues = df_all["revenue"].values

    # Normalized factors
    credit_factor = (biz.credit_score - credit_scores.min()) / (
        credit_scores.max() - credit_scores.min()
    )
    revenue_factor = (biz.revenue - revenues.min()) / (
        revenues.max() - revenues.min()
    )

    # For review strength, we approximate using collected.review_strength relative
    # to possible range in current dataset
    review_strengths = (
        df_all["google_rating"] * np.log1p(df_all["review_count"])
    ).values
    review_factor = (collected.review_strength - review_strengths.min()) / (
        review_strengths.max() - review_strengths.min()
    )

    debt_ratio_factor = collected.debt_ratio  # higher = riskier
    industry_risk = collected.industry_risk  # 0–1, higher = riskier

    # Weighted scoring
    raw_score = (
        0.35 * credit_factor
        + 0.20 * review_factor
        + 0.20 * revenue_factor
        - 0.15 * debt_ratio_factor
        - 0.10 * industry_risk
    )

    # Normalize to 0–100 with a simple linear map to [0, 1]
    risk_score = max(0.0, min(1.0, (raw_score + 0.5)))
    risk_score_0_100 = risk_score * 100.0

    risk_factors = {
        "credit_factor": float(credit_factor),
        "review_factor": float(review_factor),
        "revenue_factor": float(revenue_factor),
        "debt_ratio": float(debt_ratio_factor),
        "industry_risk": float(industry_risk),
    }

    return RiskOutput(
        business=biz,
        risk_score=float(risk_score_0_100),
        risk_factors=risk_factors,
    )


# =====================================================
# 4. RAG SETUP
#    (tiny in-memory “knowledge base”)
# =====================================================

RAG_DOCS = [
    {
        "id": "industry_food_beverage",
        "text": (
            "Food & Beverage businesses often show high seasonality and thin "
            "margins. Review quality and local reputation are strong indicators "
            "of resilience. Higher baseline risk compared with healthcare or "
            "technology."
        ),
    },
    {
        "id": "industry_retail",
        "text": (
            "Retail businesses are sensitive to consumer confidence and location "
            "foot traffic. Online reviews and repeat customers reduce perceived risk."
        ),
    },
    {
        "id": "industry_healthcare",
        "text": (
            "Healthcare businesses typically have stable demand and lower default "
            "rates, but they may be exposed to regulatory changes and insurance "
            "reimbursement risk."
        ),
    },
    {
        "id": "industry_technology",
        "text": (
            'Technology companies can scale quickly, with volatile revenue but high '
            "growth potential. Healthy cash flow and low leverage are positive signs."
        ),
    },
    {
        "id": "policy_credit_minimum",
        "text": (
            "Bank policy: Preferred applicants maintain a credit score of at least "
            "680. Applications between 640 and 680 may be considered with strong "
            "compensating factors such as low leverage and strong cash flow."
        ),
    },
    {
        "id": "policy_leverage_limit",
        "text": (
            "Bank policy: Debt-to-revenue ratios above 0.6 are considered elevated "
            "risk and typically require additional collateral or manual review."
        ),
    },
    {
        "id": "policy_review_signal",
        "text": (
            "For small businesses, sustained online customer satisfaction, reflected "
            "in a rating above 4.2 with more than 100 reviews, is treated as a "
            "strong positive qualitative factor."
        ),
    },
]


class RAGIndex:
    def __init__(self, docs: List[Dict[str, str]]):
        self.docs = docs
        self.embeddings = self._embed_docs([d["text"] for d in docs])

    def _embed_docs(self, texts: List[str]) -> np.ndarray:
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        vectors = [np.array(item.embedding, dtype=np.float32) for item in resp.data]
        return np.vstack(vectors)

    def _embed_query(self, text: str) -> np.ndarray:
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=[text],
        )
        return np.array(resp.data[0].embedding, dtype=np.float32)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        q_vec = self._embed_query(query)
        # cosine similarity
        norms_docs = np.linalg.norm(self.embeddings, axis=1) + 1e-8
        norm_q = np.linalg.norm(q_vec) + 1e-8
        sims = (self.embeddings @ q_vec) / (norms_docs * norm_q)
        top_idx = np.argsort(-sims)[:k]
        return [self.docs[i]["text"] for i in top_idx]


# Build global RAG index once
RAG_INDEX = RAGIndex(RAG_DOCS)


def rag_agent(risk_output: RiskOutput, collected: CollectorOutput) -> RAGContext:
    biz = risk_output.business
    query = (
        f"Industry: {biz.industry}. "
        f"Risk score: {risk_output.risk_score:.1f}. "
        f"Debt ratio: {collected.debt_ratio:.2f}. "
        f"Credit score: {biz.credit_score}."
    )
    chunks = RAG_INDEX.retrieve(query, k=3)
    return RAGContext(retrieved_chunks=chunks)


# =====================================================
# 5. UNDERWRITER AGENT (LLM)
# =====================================================

def _format_risk_factors(risk_factors: Dict[str, Any]) -> str:
    return "\n".join(
        f"- {key}: {value:.3f}" if isinstance(value, float) else f"- {key}: {value}"
        for key, value in risk_factors.items()
    )


def underwriter_agent(
    collected: CollectorOutput, risk_output: RiskOutput, rag_context: RAGContext
) -> UnderwriterOutput:
    biz = risk_output.business

    policy_text = "\n\n".join(rag_context.retrieved_chunks)
    risk_factor_text = _format_risk_factors(risk_output.risk_factors)

    prompt = f"""
You are a senior small-business underwriter at a retail bank.

Use the business data, engineered risk factors, and retrieved policy / industry context
to make a decision.

### Business Snapshot
Name: {biz.name}
Industry: {biz.industry}
Revenue: ${biz.revenue:,.0f}
Debt: ${biz.debt:,.0f}
Credit Score: {biz.credit_score}
Google Rating: {biz.google_rating}
Review Count: {biz.review_count}

Collector Notes: {collected.notes}

Review Strength: {collected.review_strength:.2f}
Debt Ratio: {collected.debt_ratio:.2f}
Industry Risk Score (0-1, higher = riskier): {collected.industry_risk:.2f}

### Engineered Risk Score
Overall Risk Score (0-100, higher = safer): {risk_output.risk_score:.1f}

Detailed Factors:
{risk_factor_text}

### Retrieved Context (RAG)
These are relevant snippets from internal policy and industry knowledge:

{policy_text}

### Task
1. Briefly summarize the business and risk position (2-3 sentences).
2. List key strengths (bullet points).
3. List key risks (bullet points).
4. Make a final decision: one of APPROVE, DECLINE, or REVIEW.
5. Provide a numeric confidence score between 0 and 1.

Respond in the following JSON-like structure:

SUMMARY: <text>
STRENGTHS:
- <item>
- <item>
RISKS:
- <item>
- <item>
DECISION: <APPROVE|DECLINE|REVIEW>
CONFIDENCE: <0-1 number>
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    content = response.choices[0].message.content

    # Heuristic parsing for DECISION and CONFIDENCE
    decision = "REVIEW"
    confidence = 0.5

    if "DECISION:" in content:
        try:
            after = content.split("DECISION:")[1].strip()
            first_line = after.splitlines()[0].strip()
            if "APPROVE" in first_line.upper():
                decision = "APPROVE"
            elif "DECLINE" in first_line.upper():
                decision = "DECLINE"
            else:
                decision = "REVIEW"
        except Exception:
            pass

    if "CONFIDENCE:" in content:
        try:
            after = content.split("CONFIDENCE:")[1].strip()
            first_line = after.splitlines()[0].strip()
            confidence = float(first_line)
        except Exception:
            pass

    return UnderwriterOutput(
        memo=content,
        decision=decision,
        confidence=confidence,
    )


# =====================================================
# 6. PUBLIC API FOR STREAMLIT
# =====================================================

def get_business_df() -> pd.DataFrame:
    return generate_sample_businesses()


def run_full_pipeline(row: pd.Series, df_all: pd.DataFrame) -> Dict[str, Any]:
    """
    Orchestration function used by Streamlit:
    Collector -> Risk Analyst -> RAG -> Underwriter
    """
    collected = collector_agent(row)
    risk_output = risk_analyst_agent(collected, df_all)
    rag_ctx = rag_agent(risk_output, collected)
    underwriter = underwriter_agent(collected, risk_output, rag_ctx)

    return {
        "collector": collected,
        "risk_output": risk_output,
        "rag_context": rag_ctx,
        "underwriter": underwriter,
    }
