import streamlit as st

st.set_page_config(page_title="Text Analysis")

st.title("Text Analysis App")
st.write("Use the pages in the sidebar:")
st.markdown("- **Sentiment (BERT)**: POSITIVE / NEGATIVE with confidence")
st.markdown("- **Emotion (NRCLex)**: lexicon-based emotion scores")
