# mistral7b_pdf_query_local

Build a chatbot that answers queries related to an uploaded pdf that runs locally  
  1. Use Ollama to deploy a model locally
  2. Use langchain to prep input
  3. Use langchain-community to create embeddings, clean data, and parse output
  4. Use Chroma DB as a vector store
  5. Use streamlit as an app interface  

Dependencies:
  
    langchain
    langchain-community
    streamlit
    streamlit-chat
    pypdf
    chromadb
    fastembed
    python 3.11

Recommended:

    virtualenv

To set up with a virtual environment

    pip install virtualenv
    python3.11 -m venv myenv
    source myenv/bin/activate
    pip install -r requirements.txt

See detailed instructions in original post:  
https://medium.com/@vndee.huynh/build-your-own-rag-and-run-it-locally-langchain-ollama-streamlit-181d42805895

pystemmer install had a certificate verification issue on mac, the fix is described here:  
https://stackoverflow.com/questions/52805115/certificate-verify-failed-unable-to-get-local-issuer-certificate/59692810#59692810

    cmd + space
    Install Certificates.command
