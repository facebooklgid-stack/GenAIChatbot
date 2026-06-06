import streamlit as st
import langchain
from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
##from langchain.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)


prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question.
            <context>
            {context}
            </context>
            """
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
    ]
)


def create_vector_embeddings(show_progress: bool = True):
    if "vectors" in st.session_state:
        return

    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    loader = PyPDFLoader("Flexible Pay Allowance (FPA) Policy_V1.0.pdf")  ## Data ingestion step
    page_docs = loader.load()  ## load PDF as page-level documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(page_docs)

    # Build FAISS index incrementally so we can show progress in the UI
    store = None
    total = len(final_documents)
    batch_size = 10

    if show_progress:
        spinner = st.spinner("Please wait while converting documents to vectors and building index...")
        progress_bar = st.progress(0)
        status = st.empty()
    else:
        spinner = contextmanager_noop()
        progress_bar = None
        status = None

    with spinner:
        for i in range(0, total, batch_size):
            batch = final_documents[i : i + batch_size]
            if store is None:
                store = FAISS.from_documents(batch, st.session_state.embeddings)
            else:
                # add_documents should be available on the FAISS wrapper
                try:
                    store.add_documents(batch, st.session_state.embeddings)
                except Exception:
                    # fallback: rebuild incrementally by merging indexes (rare)
                    all_docs = []
                    if store is not None:
                        # extract existing docs if possible (best-effort)
                        try:
                            all_docs = store.documents + batch
                        except Exception:
                            all_docs = final_documents[: i + batch_size]
                    else:
                        all_docs = final_documents[: i + batch_size]
                    store = FAISS.from_documents(all_docs, st.session_state.embeddings)

            if show_progress and progress_bar is not None:
                progress = int(((i + batch_size) / max(1, total)) * 100)
                progress_bar.progress(min(100, progress))
                if status is not None:
                    status.text(f"Processed {min(i + batch_size, total)}/{total} chunks")

    # finalize
    if show_progress and status is not None:
        status.text("Vector DB ready")
    st.session_state.vectors = store

if "vectors" not in st.session_state:
        create_vector_embeddings()

def contextmanager_noop():
    # simple context manager that does nothing, used when spinner not desired
    class _Noop:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    return _Noop()

st.title("INADEV Flexible Pay Allowance (FPA) Policy Chatbot")

user_prompt = st.text_input("Ask a question about the document:")

if user_prompt:
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": user_prompt})
    st.write(response["answer"])
    