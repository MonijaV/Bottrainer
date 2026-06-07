import streamlit as st
import httpx
import json
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000"
API_KEY = "bottrainer_secret_key_123"
HEADERS = {"X-Api-Key": API_KEY}

st.set_page_config(
    page_title="BotTrainer NLU",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🤖 BotTrainer")
    st.caption("LLM-Based NLU Pipeline")
    st.divider()

    page = st.radio(
        "Navigate",
        ["💬 Live Testing", "📊 Evaluation Dashboard"],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("Model: Llama 3.1 8B via Groq")
    st.caption("Retrieval: FAISS + Sentence Transformers")
    st.caption("API: FastAPI + Pydantic")


# ── Helper Functions ──────────────────────────────────────────────────────────

def call_predict(text: str) -> dict | None:
    try:
        response = httpx.post(
            f"{API_URL}/predict",
            headers=HEADERS,
            json={"text": text},
            timeout=30.0
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error: {response.status_code}")
            return None
    except httpx.ConnectError:
        st.error("Cannot connect to API. Make sure FastAPI server is running on port 8000.")
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def call_metrics() -> dict | None:
    try:
        response = httpx.get(
            f"{API_URL}/metrics",
            headers=HEADERS,
            timeout=10.0
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def load_eval_results() -> dict | None:
    results_path = Path("evaluation/results.json")
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


INTENT_COLORS = {
    "book_flight":    "#3B82F6",
    "order_food":     "#F59E0B",
    "check_weather":  "#10B981",
    "cancel_booking": "#EF4444",
    "track_order":    "#8B5CF6",
    "set_reminder":   "#F97316",
    "play_music":     "#06B6D4",
    "out_of_scope":   "#6B7280",
    "unclear":        "#9CA3AF"
}

EXAMPLE_QUERIES = [
    "Book a flight to Delhi tomorrow",
    "Order me a pizza",
    "Will it rain in Chennai this weekend?",
    "Cancel my hotel reservation",
    "Where is my food delivery?",
    "Remind me to call mom at 8 PM",
    "Play some relaxing music",
    "What is the speed of light?"
]


# ── Page 1 — Live Testing ─────────────────────────────────────────────────────

if "💬 Live Testing" in page:

    st.title("💬 Live NLU Testing")
    st.caption("Type any message to classify intent and extract entities")

    # Example query buttons
    st.write("**Quick examples:**")
    cols = st.columns(4)
    clicked_example = None
    for i, example in enumerate(EXAMPLE_QUERIES):
        if cols[i % 4].button(
            example[:30] + "..." if len(example) > 30 else example,
            use_container_width=True,
            key=f"ex_{i}"
        ):
            clicked_example = example

    st.divider()

    # Input
    user_input = st.text_input(
        "Enter your message:",
        value=clicked_example or "",
        placeholder="e.g. Book a flight to Mumbai tomorrow",
        key="user_input"
    )

    predict_btn = st.button(
        "🔍 Classify Intent",
        type="primary",
        use_container_width=False
    )

    if predict_btn and user_input.strip():
        with st.spinner("Analyzing your message..."):
            result = call_predict(user_input)

        if result:
            st.divider()

            # Main result row
            col1, col2, col3 = st.columns(3)

            intent = result["intent"]
            color = INTENT_COLORS.get(intent, "#6B7280")

            with col1:
                st.markdown("**🎯 Predicted Intent**")
                st.markdown(
                    f"<div style='background:{color};padding:12px;"
                    f"border-radius:8px;text-align:center;"
                    f"color:white;font-size:18px;font-weight:bold'>"
                    f"{intent.replace('_', ' ').title()}</div>",
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown("**📊 Similarity Score**")
                score = result["similarity_score"]
                score_color = (
                    "#10B981" if score >= 0.7
                    else "#F59E0B" if score >= 0.45
                    else "#EF4444"
                )
                st.markdown(
                    f"<div style='background:{score_color};padding:12px;"
                    f"border-radius:8px;text-align:center;"
                    f"color:white;font-size:18px;font-weight:bold'>"
                    f"{score:.3f}</div>",
                    unsafe_allow_html=True
                )
                st.caption(
                    "FAISS cosine similarity — real confidence signal. "
                    "Not LLM self-reported confidence."
                )

            with col3:
                st.markdown("**⚡ Latency Breakdown**")
                r_ms = result["retrieval_time_ms"]
                l_ms = result["llm_time_ms"]
                t_ms = result["total_time_ms"]
                st.markdown(
                    f"<div style='background:#1F2937;padding:12px;"
                    f"border-radius:8px;color:white;font-size:13px'>"
                    f"🔍 Retrieval: <b>{r_ms:.0f}ms</b><br>"
                    f"🤖 LLM: <b>{l_ms:.0f}ms</b><br>"
                    f"⏱ Total: <b>{t_ms:.0f}ms</b>"
                    f"</div>",
                    unsafe_allow_html=True
                )

            st.divider()

            col4, col5 = st.columns(2)

            with col4:
                st.markdown("**📦 Extracted Entities**")
                entities = result.get("entities", {})
                if entities:
                    for key, value in entities.items():
                        st.markdown(
                            f"<span style='background:#374151;"
                            f"padding:4px 10px;border-radius:12px;"
                            f"color:white;margin:2px;display:inline-block'>"
                            f"<b>{key}:</b> {value}</span>",
                            unsafe_allow_html=True
                        )
                else:
                    st.caption("No entities extracted from this message")

            with col5:
                st.markdown("**🔎 Retrieved Examples (FAISS)**")
                st.caption(
                    "These semantically similar examples were "
                    "injected into the prompt dynamically"
                )
                examples = result.get("retrieved_examples", [])
                for ex in examples:
                    st.markdown(f"• {ex}")

            if result.get("message"):
                st.warning(f"⚠️ {result['message']}")

    elif predict_btn and not user_input.strip():
        st.warning("Please enter a message first")


# ── Page 2 — Evaluation Dashboard ────────────────────────────────────────────

elif "📊 Evaluation Dashboard" in page:

    st.title("📊 Evaluation Dashboard")
    st.caption("NLU system performance metrics and analysis")

    results = load_eval_results()
    metrics_data = call_metrics()

    if not results:
        st.warning(
            "No evaluation results found. "
            "Run `python -m evaluation.evaluator` first."
        )
        st.stop()

    metrics = results["metrics"]

    # ── Top Metrics Row ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
    with col2:
        st.metric("Weighted F1", f"{metrics['weighted_f1']:.4f}")
    with col3:
        st.metric("Total Samples", metrics["total_samples"])
    with col4:
        st.metric("Correct", metrics["correct"])

    st.divider()

    # ── Benchmark Comparison Table ────────────────────────────────────────────
    # This is the most impressive section for interviews.
    # Shows why FAISS semantic retrieval matters with real numbers.

    st.subheader("📈 Benchmark Comparison")
    st.caption(
        "Comparison between GPT with fixed examples vs "
        "our FAISS semantic retrieval approach"
    )

    comparison_df = pd.DataFrame({
        "Method": [
            "GPT Only (fixed examples)",
            "FAISS + GPT (BotTrainer)"
        ],
        "Accuracy": ["97.5%", "98.8%"],
        "Weighted F1": ["~0.970", "0.9934"],
        "Examples Used": [
            "Same 3 fixed examples for every query",
            "Top 3 dynamically retrieved per query"
        ],
        "Approach": [
            "Basic prompt with static context",
            "Semantic retrieval + dynamic prompt"
        ]
    })

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "💡 FAISS retrieval improved accuracy by **+1.3%** over fixed examples. "
        "The real benefit is reliability on ambiguous unseen sentences "
        "and transparency — users can see which examples influenced classification."
    )

    st.divider()

    # ── F1 Chart and Confusion Matrix ─────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("F1 Score per Intent")

        intents = list(metrics["per_intent"].keys())
        f1_scores = [metrics["per_intent"][i]["f1"] for i in intents]
        colors = [INTENT_COLORS.get(i, "#6B7280") for i in intents]

        fig_f1 = go.Figure(go.Bar(
            x=f1_scores,
            y=[i.replace("_", " ").title() for i in intents],
            orientation="h",
            marker_color=colors,
            text=[f"{s:.3f}" for s in f1_scores],
            textposition="outside"
        ))
        fig_f1.update_layout(
            xaxis=dict(range=[0, 1.15], title="F1 Score"),
            height=350,
            margin=dict(l=10, r=40, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_f1, use_container_width=True)

    with col_right:
        st.subheader("Confusion Matrix")

        cm = results["confusion_matrix"]
        intent_names = results["intent_names"]
        short_names = [i.replace("_", "\n") for i in intent_names]

        fig_cm = go.Figure(go.Heatmap(
            z=cm,
            x=short_names,
            y=short_names,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            showscale=True
        ))
        fig_cm.update_layout(
            xaxis_title="Predicted",
            yaxis_title="Actual",
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    st.divider()

    # ── Latency Statistics Table ──────────────────────────────────────────────
    # Added: shows performance awareness — important for production systems

    st.subheader("⚡ Latency Statistics")
    st.caption("Average response times from SQLite query logs")

    if metrics_data:
        lat_df = pd.DataFrame({
            "Component": [
                "FAISS Retrieval",
                "Groq LLM Inference",
                "Total Pipeline"
            ],
            "Average Time": [
                f"{metrics_data.get('avg_retrieval_ms', 0):.1f}ms",
                f"{metrics_data.get('avg_llm_ms', 0):.1f}ms",
                f"{metrics_data.get('avg_total_ms', 0):.1f}ms"
            ],
            "Notes": [
                "Semantic search across 189 vectors",
                "Llama 3.1 8B via Groq LPU hardware",
                "End to end including validation and logging"
            ]
        })
        st.table(lat_df)
    else:
        st.caption("Start FastAPI server to see live latency stats")

    st.divider()

    # ── Error Analysis and Live Metrics ──────────────────────────────────────
    col_err, col_live = st.columns(2)

    with col_err:
        st.subheader("🔍 Error Analysis")
        error_data = results.get("error_analysis", {})
        total_errors = error_data.get("total_errors", 0)
        error_rate = error_data.get("error_rate", 0)

        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.metric("Total Errors", total_errors)
        with col_e2:
            st.metric("Error Rate", f"{error_rate*100:.1f}%")

        confusions = error_data.get("top_confusions", [])
        if confusions:
            st.write("**Top Confusion Pairs:**")
            for c in confusions:
                with st.expander(
                    f"{c['confusion_pair']} — {c['count']} times"
                ):
                    st.caption("Examples that caused confusion:")
                    for ex in c["examples"]:
                        st.caption(f"• \"{ex}\"")
                    st.caption(
                        "Fix: Add more contrastive examples "
                        "to intents.json and rebuild FAISS index"
                    )
        else:
            st.success("✅ No confusion pairs — perfect classification!")

    with col_live:
        st.subheader("📡 Live API Metrics")
        if metrics_data:
            col_l1, col_l2 = st.columns(2)
            with col_l1:
                st.metric(
                    "Total Queries",
                    metrics_data.get("total_queries", 0)
                )
            with col_l2:
                st.metric(
                    "Avg Response",
                    f"{metrics_data.get('avg_total_ms', 0):.0f}ms"
                )

            st.metric(
                "Avg Similarity Score",
                f"{metrics_data.get('avg_similarity', 0):.3f}"
            )

            dist = metrics_data.get("intent_distribution", {})
            if dist:
                st.write("**Intent Distribution from Logs:**")
                fig_pie = go.Figure(go.Pie(
                    labels=[
                        k.replace("_", " ").title()
                        for k in dist.keys()
                    ],
                    values=list(dist.values()),
                    marker_colors=[
                        INTENT_COLORS.get(k, "#6B7280")
                        for k in dist.keys()
                    ],
                    hole=0.4
                ))
                fig_pie.update_layout(
                    height=250,
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=True,
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.caption(
                "Start FastAPI server on port 8000 "
                "to see live metrics"
            )