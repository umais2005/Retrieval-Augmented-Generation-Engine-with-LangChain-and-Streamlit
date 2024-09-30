import os, tempfile
from pathlib import Path

from langchain_huggingface.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("groq_api_key")
os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")

TMP_DIR = "./data"
LOCAL_VECTOR_STORE_DIR = "./data/vector_store"

st.set_page_config(page_title="AI")
st.title("Arabic ai chatbot")


def load_documents():
    loader = DirectoryLoader("./data", glob='**/*.pdf')
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):

    embeddings=HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vectordb = FAISS.from_documents(texts, embedding=embeddings)
    retriever = vectordb.as_retriever()
    return retriever

def init_llm(groq=True):
    if groq:
        llm = ChatGroq(
            api_key=groq_api_key,
            model="llama3-8b-8192",
            temperature=0.0)
    return llm


def query_llm(query,retriever):
    system_prompt = "You are a helpful assistant that answers questions clearly and concisely."
    # Combine system prompt and chat history
    chat_history = st.session_state.messages
    chat_with_system_prompt = [(system_prompt, "")] + chat_history + [(query, "")]  # prepend system prompt
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=init_llm(),
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': query, 'chat_history': chat_with_system_prompt})
    result = result['answer']

    st.session_state.messages.append((query, result))
    return result


def process_documents():
    if not st.session_state.source_docs:
        st.warning(f"Please upload the document")
    else:
        documents=[]
        for uploaded_file in st.session_state.source_docs:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)
        texts = split_documents(documents)
        st.session_state.retriever = embeddings_on_local_vectordb(texts)

def boot():

    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)

    #
    st.button("Submit Documents", on_click=process_documents)
    #
    if "messages" not in st.session_state:
        st.session_state.messages = []
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm(query,st.session_state.retriever)
        st.chat_message("ai").write(response)

if __name__ == '__main__':
    #
    boot()
    