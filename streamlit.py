import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="FinTech RAG", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []


def ask_api(question):

    try:
        r = requests.post(
            f"{API_URL}/ask",
            json={"question": question}
        )

        if r.status_code == 200:
            return r.json()

        return {"answer": f"API Error: {r.text}", "sources": []}

    except Exception as e:
        return {"answer": f"Connection Error: {str(e)}", "sources": []}


st.title("💬 FinTech RAG Assistant")

# ---------- Display Messages ----------

for msg in st.session_state.messages:

    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])

    else:

        content = msg["content"]

        if isinstance(content, dict) and "answer" in content:

            st.chat_message("assistant").write(content["answer"])

            if content.get("sources"):
                with st.expander("Sources"):
                    for s in content["sources"]:
                        st.write(s)

        else:
            st.chat_message("assistant").write(content)

# ---------- Chat Input ----------

prompt = st.chat_input("Ask something")

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.spinner("Thinking..."):
        response = ask_api(prompt)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })

    st.rerun()
