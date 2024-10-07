import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import os
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_cohere import CohereEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import dotenv

dotenv.load_dotenv()

# Load necessary environment variables
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

# Set up LLM and embeddings
llm = ChatCohere(model="command-r",temperature=0.2)
embeddings = CohereEmbeddings(model="embed-english-light-v3.0")

# Load FAISS databases
vector_database = os.path.join(os.getcwd(), "FOBOH_index")
db = FAISS.load_local(folder_path=vector_database, embeddings=embeddings, index_name="index", allow_dangerous_deserialization=True)

catalogue_database = os.path.join(os.getcwd(), "FOBOH_PROD_CATALOGUE_index")
db_catalogue = FAISS.load_local(folder_path=catalogue_database, embeddings=embeddings, index_name="index", allow_dangerous_deserialization=True)

# Load catalogue PDF
catalogue = ''
pdf_reader = PdfReader(os.path.join(os.getcwd(), "prod_catalogue.pdf"))
for page in pdf_reader.pages:
    catalogue += page.extract_text()

# Define the prompt template
prompt_template = """ Think step by step.
   You are a sales assistant, you need to help the sales person from a food supplier company to give smarter product suggestions.
    You need to provide suggestions based on the sales person's product catalogue.
    The sales person's product catalogue contains all products that he can sell. It is shown in the following as catalogue,
    You need to find the most suitable suggestion from the product catalogue according to restaurant menus.
    The restaurant menus are shown in the following as context.
    Use the following pieces of retrieved context to answer the question based on catalogue information:  
    
    catalogue:
    {catalogue}

    context: {context}

    question : {question}

    Answer:
"""

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Set up the Streamlit interface
st.sidebar.image(os.path.join(os.getcwd(), "logo.png"), use_column_width=True)

side_bar_message = """
"""

st.sidebar.markdown(side_bar_message)

initial_message = """
    Hi there! As Sales virtual assistant Bot 🤖, I have access to the latest warehouse products and materials. Please ask your question:
"""

# Initialize session state for messages
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": initial_message}]

# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Add chat input
if prompt := st.chat_input("Write your question for your selected restaurant, please:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# Clear chat button
st.button('Clear Chat', on_click=clear_chat_history)

# Text input for user to select a restaurant
user_selected_restaurant = st.text_input("Enter the name of a restaurant for processing:", "")

# Store the selected restaurant in session state
if "selected_restaurant" not in st.session_state:
    st.session_state["selected_restaurant"] = ""

# Update restaurant selection and reset chat if a new restaurant is selected
if user_selected_restaurant and user_selected_restaurant != st.session_state["selected_restaurant"]:
    st.session_state["selected_restaurant"] = user_selected_restaurant
    st.session_state.messages.append({"role": "assistant", "content": f"Restaurant changed to {user_selected_restaurant}. How can I assist you?"})

# Function to generate responses
def get_response(question):
    selected_restaurant = st.session_state["selected_restaurant"]
    
    if not selected_restaurant:
        return "Please select a restaurant first."

    FAISS_retriever = db.as_retriever(metadata={'name': f'{selected_restaurant}'}, top_k=5)
    
    template = prompt_template
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {
            "context": itemgetter("question") | FAISS_retriever | format_docs,
            "catalogue": itemgetter("catalogue"),
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke({
        "question": f"For this restaurant {selected_restaurant}, {question}",
        "catalogue": catalogue
    })

    return response

# Automatically generate responses if the last message is from the user
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    with st.chat_message("assistant"):
        with st.spinner("Hold on, I'm fetching the latest response..."):
            user_message = st.session_state.messages[-1]["content"]
            response = get_response(user_message)
            placeholder = st.empty()
            full_response = response
            placeholder.markdown(full_response)

    # Append assistant's response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
