import os
import requests
import streamlit as st

st.set_page_config(
    page_title="Document Verifier",
    page_icon="📄",
    layout="centered"
)

API_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000"
)


st.title("📄 Document Verifier")
st.caption("Upload PDFs and ask questions using RAG.")


# -------------------------
# Backend Health Check
# -------------------------

try:
    health = requests.get(
        f"{API_URL}/health",
        timeout=15
    )

    if health.status_code == 200:
        st.success("Backend connected!")
    else:
        st.error("Backend is not responding.")

except requests.RequestException:
    st.error("Cannot connect to backend.")


# -------------------------
# PDF Upload
# -------------------------

st.header("Upload PDF")

uploaded_file = st.file_uploader(
    "Choose a PDF",
    type=["pdf"]
)

if st.button("Upload and Ingest"):

    if uploaded_file is None:
        st.warning("Please select a PDF first.")

    else:
        with st.spinner("Uploading PDF..."):

            try:
                response = requests.post(
                    f"{API_URL}/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    },
                    timeout=60
                )

                response.raise_for_status()

                result = response.json()

                st.success(
                    "PDF ingestion event submitted!"
                )

                st.json(result)

            except requests.RequestException as e:
                st.error(f"Upload failed: {e}")


st.divider()


# -------------------------
# RAG Question Answering
# -------------------------

st.header("Ask Questions")

with st.form("query_form"):

    question = st.text_input(
        "Enter your question"
    )

    top_k = st.number_input(
        "Number of chunks",
        min_value=1,
        max_value=20,
        value=5
    )

    submitted = st.form_submit_button(
        "Generate Answer"
    )


if submitted:

    if not question.strip():
        st.warning("Enter a question.")

    else:

        with st.spinner("Searching documents and generating answer..."):

            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={
                        "question": question,
                        "top_k": int(top_k)
                    },
                    timeout=180
                )

                response.raise_for_status()

                result = response.json()

                st.subheader("Answer")

                st.write(result["answer"])

                st.subheader("Sources")

                for source in result.get("sources", []):
                    st.write(f"- {source}")

                st.caption(
                    f"Retrieved contexts: "
                    f"{result['num_contexts']}"
                )

            except requests.RequestException as e:
                st.error(f"Query failed: {e}")
                
