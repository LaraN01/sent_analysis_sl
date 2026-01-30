import streamlit as st
import nltk
from nrclex import NRCLex

st.set_page_config(page_title="Emotion (NRCLex)", page_icon="🎭", layout="centered")
st.title("🎭 Emotion Analysis (NRCLex)")
st.caption("Lexicon-based emotion detection (no ML model download).")

@st.cache_resource
def ensure_nltk():
    # NRCLex uses NLTK tokenization; this makes it work on Streamlit Cloud.
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    return True

ensure_nltk()

text = st.text_area("Enter text", height=160, placeholder="I feel excited but also a bit nervous...")

col1, col2 = st.columns([1, 1])
analyze = col1.button("Analyze emotions", type="primary", use_container_width=True)
clear = col2.button("Clear", use_container_width=True)

if clear:
    st.session_state["emo_result"] = None
    st.rerun()

if analyze:
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            emo = NRCLex(text)
            # affect_frequencies gives normalized scores per emotion
            result = {
                "top_emotions": emo.top_emotions,           # list of tuples
                "affect_frequencies": emo.affect_frequencies, # dict
                "raw_emotions": emo.raw_emotion_scores,     # dict counts
            }
            st.session_state["emo_result"] = result

res = st.session_state.get("emo_result")

if res:
    st.subheader("Top emotions")
    if res["top_emotions"]:
        for emotion, score in res["top_emotions"]:
            st.write(f"- **{emotion}**: {score:.3f}")
    else:
        st.write("No strong emotion signals detected.")

    st.subheader("Emotion frequencies")
    st.write(res["affect_frequencies"])

    with st.expander("Raw scores"):
        st.json(res)
