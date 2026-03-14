from __future__ import annotations

import streamlit as st

from context_lab.config import Config
from context_lab.data import load_qmsum_examples
from context_lab.pipeline import ContextLabPipeline


@st.cache_resource(show_spinner=False)
def load_pipeline(config_path: str) -> ContextLabPipeline:
    return ContextLabPipeline(Config.from_yaml(config_path))


@st.cache_data(show_spinner=False)
def load_examples(max_samples: int) -> list:
    return load_qmsum_examples(split="validation", max_samples=max_samples)


st.set_page_config(page_title="Word Copilot Context Lab", layout="wide")
st.title("Word Copilot Context Lab")
st.caption("Compare a chunk-only baseline with a context-aware long-document assistant.")

config_path = st.sidebar.text_input("Config path", value="configs/default.yaml")
max_samples = st.sidebar.slider("Loaded examples", min_value=5, max_value=100, value=20, step=5)
example_index = st.sidebar.number_input("Example index", min_value=0, max_value=max_samples - 1, value=0)

examples = load_examples(max_samples=max_samples)
example = examples[int(example_index)]
pipeline = load_pipeline(config_path)

st.subheader("Query")
st.write(example.query)

with st.expander("Reference answer"):
    st.write(example.reference)

with st.expander("Document preview"):
    st.write(example.document[:4000] + ("..." if len(example.document) > 4000 else ""))

if st.button("Run comparison"):
    baseline = pipeline.run_example(example, system="baseline")
    context_aware = pipeline.run_example(example, system="context_aware")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Baseline")
        st.write(baseline.prediction)
        st.caption(f"Latency: {baseline.latency_sec:.2f}s")
        st.text_area("Baseline context", baseline.packed_context, height=300)

    with col2:
        st.markdown("### Context-aware")
        st.write(context_aware.prediction)
        st.caption(f"Latency: {context_aware.latency_sec:.2f}s")
        st.text_area("Global summary", context_aware.global_summary, height=150)
        st.text_area("Packed context", context_aware.packed_context, height=200)
