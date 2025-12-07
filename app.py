import streamlit as st
import pandas as pd

from underwriting_agents_rag import (
    get_business_df,
    run_full_pipeline,
)

st.set_page_config(page_title="AI Underwriting (Agents + RAG)", layout="wide")

st.title("🏦 AI Underwriting Demo – Multi‑Agent + RAG")

st.write(
    "This demo simulates a retail bank's underwriting process using multiple "
    "LLM agents and a tiny retrieval‑augmented (RAG) knowledge base for "
    "industry and policy context."
)

# Load sample data once
@st.cache_data
def load_data():
    return get_business_df()

df = load_data()

st.sidebar.header("Business Selector")
selected_name = st.sidebar.selectbox(
    "Choose a business:", df["name"].tolist()
)

row = df[df["name"] == selected_name].iloc[0]

# Layout columns
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📊 Business Snapshot")
    st.markdown(f"**Name:** {row['name']}")
    st.markdown(f"**Industry:** {row['industry']}")
    st.markdown(f"**Revenue:** ${row['revenue']:,.0f}")
    st.markdown(f"**Debt:** ${row['debt']:,.0f}")
    st.markdown(f"**Credit Score:** {row['credit_score']}")
    st.markdown(f"**Google Rating:** {row['google_rating']}")
    st.markdown(f"**Review Count:** {row['review_count']}")

    st.subheader("🔎 Raw Data Row")
    st.dataframe(pd.DataFrame([row]))

with col2:
    st.subheader("⚙️ Run Underwriting Pipeline")

    if st.button("Run Multi‑Agent + RAG Analysis"):
        with st.spinner("Running agents and retrieving policy context..."):
            result = run_full_pipeline(row, df)

        collector = result["collector"]
        risk_output = result["risk_output"]
        rag_ctx = result["rag_context"]
        underwriter = result["underwriter"]

        # ---- Collector / Feature Engineering ----
        st.markdown("### 1️⃣ Collector Agent (Signals & Features)")
        st.markdown(f"- **Review Strength:** {collector.review_strength:.2f}")
        st.markdown(f"- **Debt Ratio:** {collector.debt_ratio:.2f}")
        st.markdown(
            f"- **Industry Risk (0–1, higher = riskier):** {collector.industry_risk:.2f}"
        )
        st.markdown(f"**Notes:** {collector.notes}")

        # ---- Risk Analyst ----
        st.markdown("### 2️⃣ Risk Analyst Agent (Score & Factors)")
        st.markdown(
            f"**Overall Risk Score (0–100, higher = safer):** "
            f"{risk_output.risk_score:.1f}"
        )

        factors_df = pd.DataFrame(
            list(risk_output.risk_factors.items()),
            columns=["Factor", "Value"],
        )
        st.dataframe(factors_df, hide_index=True)

        # ---- RAG Context ----
        st.markdown("### 3️⃣ RAG – Retrieved Policy & Industry Context")
        for i, chunk in enumerate(rag_ctx.retrieved_chunks, start=1):
            st.markdown(f"**Snippet {i}:**")
            st.info(chunk)

        # ---- Underwriter Memo ----
        st.markdown("### 4️⃣ Final Underwriter Agent (Decision & Memo)")
        st.markdown(
            f"**Decision:** :blue[{underwriter.decision}]  "
            f"**Confidence:** {underwriter.confidence:.2f}"
        )

        st.markdown("**Underwriting Memo:**")
        st.write(underwriter.memo)
    else:
        st.info("Click **Run Multi‑Agent + RAG Analysis** to generate a decision.")
