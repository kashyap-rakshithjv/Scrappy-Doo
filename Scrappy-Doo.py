import streamlit as st
from langchain import LLMChain, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
#from langchain.vectorstores import Pinecone
#from pinecone import Pinecone
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os

#os.system('pip install beautifulsoup4')

#os.environ['PINECONE_API_KEY'] = "pcsk_5WxMYQ_3MtZmZK8kdTitc7B8qvw7HfcD8W8FnDEKRyevPxnYAdn9TVDa1SaAPkXKb68aTK"
os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]

def load_website(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    return data

# Streamlit UI
st.title("Scrappy-Doo")
st.write("Extract data from any website with ease!")

link = st.text_input("URL:")

if st.button("Fetch"):
    if link:
        with st.spinner("Loading website data..."):
            website_data = load_website(link)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                length_function=len
            )
            split_docs = text_splitter.split_documents(website_data)
            embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            doc_embeds = embed_model.embed_documents(link)
            
            #index_name = "scrappy-doo"
            #vectorstore = Pinecone.from_documents(split_docs, embed_model, index_name=index_name)
        
            vectorstore = Chroma.from_documents(split_docs, embed_model)
            
            llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
            
            qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
                memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            )
        st.session_state.qa = qa
        st.session_state.vectorstore = vectorstore
        
        st.success("Website data loaded successfully! You can now ask questions about the website.")
    else:
        st.warning("Please enter a URL before clicking Fetch!")

# Check if `qa` exists in session state
if "qa" in st.session_state:
    query = st.text_input("Ask a question about the website:")
    if st.button("Ask"):
        if query:
            # Use the `qa` object from session state
            response = st.session_state.qa({"query": query})
            st.subheader("AI Response:")
            st.write(response.get('result', response))
        else:
            st.warning("Please enter a question before clicking Ask!")
