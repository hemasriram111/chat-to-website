import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import ollama
from pydantic import BaseModel
from typing import Optional, List
from langchain.schema import LLMResult, Generation

# Define custom Ollama embeddings
class OllamaEmbeddings(Embeddings):
    def __init__(self, model_name="llama2"):
        self.model_name = model_name

    def embed_documents(self, texts):
        return [ollama.embeddings(model=self.model_name, prompt=text)["embedding"] for text in texts]

    def embed_query(self, text):
        return ollama.embeddings(model=self.model_name, prompt=text)["embedding"]

# Define custom LLM wrapper
class OllamaLLM(BaseModel):
    model_name: str = "llama2"

    def __call__(self, prompt: str) -> str:
        return ollama.generate(model=self.model_name, prompt=prompt)["response"]

# Define pydantic model for input validation
class GenerateRequest(BaseModel):
    prompt: str

def get_vectorstore_from_url(url):
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    vector_store = FAISS.from_documents(document_chunks, OllamaEmbeddings())
    return vector_store

def get_context_retriever_chain(vector_store):
    llm = OllamaLLM()
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain):
    llm = OllamaLLM()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    # Validate input with Pydantic model, ensuring the input is a valid string
    try:
        # Ensure the input is a string before passing to GenerateRequest
        if isinstance(user_input, str):
            request = GenerateRequest(prompt=user_input)
        else:
            # If the input is not a string (e.g., a complex object), extract the string
            request = GenerateRequest(prompt=str(user_input))  # or handle custom extraction if needed
    except Exception as e:
        return f"Error: {str(e)}"

    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    # Directly invoke the chain with necessary inputs
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    # Extract the string from the response (assuming the response is a dictionary)
    return response.get('answer', 'Sorry, I could not find an answer.')

st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="Hello, I am a bot. How can I help you?")]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)
    
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))
    
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
