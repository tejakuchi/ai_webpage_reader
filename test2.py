
import os
import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# Fetch OpenAI API key from environment variable
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    st.error("OpenAI API key not found. Please set the 'OPENAI_API_KEY' environment variable.")
else:
    # Set the environment variable for OpenAI API key
    os.environ['OPENAI_API_KEY'] = openai_api_key

    # Streamlit app title
    st.title("Conversational AI with Webpage Data")

    # Input for the webpage URL
    webpage_url = st.text_input("Enter the Webpage URL", "")

    if webpage_url:
        try:
            # Load documents from the specified webpage
            st.write("Fetching and processing the webpage content...")
            loader = WebBaseLoader(webpage_url)
            documents = loader.load()

            # Split the documents into smaller, contextually meaningful chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,  # Adding a slight overlap to maintain context between chunks
                #length_function=len,
                #separators=["\n\n", "\n", " ", ""]
            )
            splits = text_splitter.split_documents(documents)

            # Create embeddings using OpenAI and store them in a Chroma vector store
            embedding = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)

            # Set up a retriever from the vector store
            retriever = vectorstore.as_retriever()

            # Set up conversation memory to track chat history
            memory_store = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            # Create the conversational retrieval chain using the ChatOpenAI model
            chatQA = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(temperature=0.1, model_name='gpt-3.5-turbo'),
                retriever=retriever,
                memory=memory_store
            )

            st.success("Webpage content processed successfully!")

            # Text input for user queries
            query = st.text_input("Ask a question about the webpage content", "")

            if query:
                # Get the response from the conversational AI system
                with st.spinner("Generating response..."):
                    response = chatQA({"question": query})
                    st.write(response["answer"])

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Streamlit app footer
    st.markdown("---")
    st.markdown("Developed with [LangChain](https://github.com/hwchase17/langchain) and [Streamlit](https://streamlit.io/)")

