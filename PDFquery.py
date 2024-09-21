import os
import tempfile
import streamlit as st
from streamlit_chat import message
from ragPDF import QueryPDF
from time import time

# Implemnted from https://medium.com/@vndee.huynh/build-your-own-rag-and-run-it-locally-langchain-ollama-streamlit-181d42805895
# to start, in cli: ollama pull mistral
# ollama list
# streamlit run app.py

# Give our chatbot page a title
st.cache_data.clear()
st.set_page_config(page_title="QueryPDF")

def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        t1 = time()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)
        
        exec_time = '\n Elapsed Time: ' + str(round(time() - t1,1)) + 's'
        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text + exec_time, False))

def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingest(file_path)
        os.remove(file_path)

def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = QueryPDF()

    st.header("QueryPDF")

    st.subheader("Upload a document")
    st.file_uploader(
            "Upload document",
            type = ["pdf"],
            key = "file_uploader",
            on_change = read_and_save_file,
            label_visibility = "collapsed",
            accept_multiple_files = True,
    )

    st.session_state["ingestion_spinner"] = st.empty()
    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()