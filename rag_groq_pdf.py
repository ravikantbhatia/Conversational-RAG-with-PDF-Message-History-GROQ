# Import required libraries
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_chroma import Chroma
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_classic.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os

load_dotenv()

# App title
st.title("Conversation RAG with PDF+ Message History GROQ")

# Get API keys from user
api_key=st.text_input("Provide with the GROQ API Key",type='password')
openai_api_key=st.text_input("Provide with the OPENAI API Key",type='password')

if api_key and openai_api_key:

    # Initialize Groq LLM
    model =ChatGroq(model_name='openai/gpt-oss-20b',groq_api_key=api_key)

    # Session ID for conversation history
    session_id=st.text_input("Please Provide Your Session ID: ",value='default-session')

    if 'store' not in st.session_state:
        st.session_state.store={}

    # Upload PDF
    uploaded_file=st.file_uploader("Upload Your PDFs: ",type='pdf')

    if uploaded_file:
        temp_pdf="./temporary.pdf"

        # Save uploaded PDF temporarily
        with open(temp_pdf,'wb') as f:
            f.write(uploaded_file.getvalue())

        # Load and split PDF
        loader=PyPDFLoader(temp_pdf)
        documents=loader.load()
        st.write("documents loaded:",len(documents))

        splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
        splits=splitter.split_documents(documents)
        st.write("Chunks created:",len(splits))

        # Create embeddings and vector store
        vector=Chroma.from_documents(
            splits,
            embedding=OpenAIEmbeddings(api_key=openai_api_key),
            persist_directory='./chroma_datab'
        )
        retriever=vector.as_retriever()

        # Create history-aware retriever
        contextualize_system_prompt='''Given a chat history and the latest
        user question ,reformulate the question to make it standalone.
        Do not answer
        ,only rephrase.'''

        contextualize_prompt=ChatPromptTemplate.from_messages([
            ('system',contextualize_system_prompt),
            MessagesPlaceholder('chat_history'),
        ('human','{input}')])

        history_aware_retriever=create_history_aware_retriever(
            model,retriever,contextualize_prompt
        )

        # Prompt for answering questions from PDF context
        system_prompt='''You are an helpful AI assistant.
        for Question answering ,
        Answer from the provided context only. If unsure,Say I Dont know
        context
        {context}
        '''

        qa_prompt=ChatPromptTemplate.from_messages([
            ('system',system_prompt),
            MessagesPlaceholder('chat_history'),
        ('human','{input}')])

        # Create RAG chain
        document_chain=create_stuff_documents_chain(model,qa_prompt)
        retrieval_chain=create_retrieval_chain(
            history_aware_retriever,document_chain
        )

        # Manage conversation history
        def get_session_history(session_id):
            if 'store' not in st.session_state:
                st.session_state.store = {}

            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()

            return st.session_state.store[session_id]

        # Add conversation history to RAG chain
        conversational_rag_chain=RunnableWithMessageHistory(
            retrieval_chain,
            get_session_history,
            input_messages_key='input',
            history_messages_key='chat_history',
            output_messages_key='answer'
        )

        # User question
        user_input=st.text_input("Ask a Question about Your PDF:")

        if user_input:
            session_history=get_session_history(session_id)

            # Generate answer
            response=conversational_rag_chain.invoke(
                {'input':user_input},
                config={'configurable':{'session_id':session_id}}
            )

            st.subheader("Assistant Answer:")
            st.write(response['answer'])

            # Display conversation history
            with st.expander("Chat History"):
                st.write(session_history.messages)

else:
    st.warning("Please Enter Both GROQ and OPENAI API KEYS to continue")




