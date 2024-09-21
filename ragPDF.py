from langchain_community.vectorstores import Chroma #open source vector database
from langchain_community.chat_models import ChatOllama #runs open-source LLM locally
from langchain_community.embeddings import FastEmbedEmbeddings #CPU-first designed embeddings model by Qdrant
from langchain.schema.output_parser import StrOutputParser #Parses llm output into a string
from langchain_community.document_loaders import PyPDFLoader #PDF document parser
from langchain.text_splitter import RecursiveCharacterTextSplitter #Good generic text splitter that tries to keep context together
from langchain.schema.runnable import RunnablePassthrough # TODO more clarity on this function
from langchain.prompts import PromptTemplate #Lets users create custom Q/A templates
from langchain_community.vectorstores.utils import filter_complex_metadata #Remove unsupported metadata types

# Implemnted from https://medium.com/@vndee.huynh/build-your-own-rag-and-run-it-locally-langchain-ollama-streamlit-181d42805895
# to start, in cli: ollama pull mistral
# ollama list
# streamlit run app.py

class QueryPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        # set the model
        self.model = ChatOllama(model="mistral")
        # break the text into chunks (!!experiment with chunk size/overlap)
        # TODO experiment with chunk size and overlap
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        # set up a prompt template to be used
        # https://smith.langchain.com/hub/rlm/rag-prompt-mistral
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise. [/INST] <s>
            [INST] Question: {question}
            Context: {context}
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        # load the document
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        # break up the document into meaningful chunks
        chunks = self.text_splitter.split_documents(docs)
        # remove unsupported metadata
        chunks = filter_complex_metadata(chunks)

        # put the document chunks into the database and create text embeddings
        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        # set up a simlarity retrieval mechanisms
        # TODO experiment with search types and kwargs (Returns top 3 with highest score above 0.5)
        # https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/vectorstore/
        self.retriever = vector_store.as_retriever(
            search_types = "similarity_score_threshold",
            search_kwargs = {"k": 3},
        )
        # TODO explain why this is formatted as such
        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())
    
    # Helper method to filter no-pdf loaded errors and invoke the query
    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."
        return self.chain.invoke(query)
        
    # Helper method to clear results of last query when a new PDF is loaded
    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None