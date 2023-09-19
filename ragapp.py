from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import streamlit as st
from watsonxlangchain import LangChainInterface

creds = {
    'apikey':'YOUR API KEY HERE', 
    'url': 'https://us-south.ml.cloud.ibm.com'
}
llm = LangChainInterface(credentials=creds, model='meta-llama/llama-2-70b-chat', 
params = {'decoding_method':'sample', 'max_new_tokens':200, 'temperature':0.5}, 
project_id='YOUR PROJECT ID HERE')


@st.cache_resource
def load_pdf(): 
    pdf_name = 'what is generative ai.pdf'
    loaders = [PyPDFLoader(pdf_name)]

    index = VectorstoreIndexCreator(
        embedding = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'), 
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_loaders(loaders)

    return index

index = load_pdf()

chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=index.vectorstore.as_retriever(), input_key='question')

st.title('Ask Watsonx 🤖')

if 'messages' not in st.session_state: 
    st.session_state.messages = [] 

for message in st.session_state.messages: 
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('Pass Your Prompt here')

if prompt: 
    st.chat_message('user').markdown(prompt)
    st.session_state.messages.append({'role':'user', 'content':prompt})

    response = chain.run(prompt)

    st.chat_message('assistant').markdown(response)
    st.session_state.messages.append({'role':'assistant', 'content':response})