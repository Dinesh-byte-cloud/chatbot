import os
import tempfile
import io
from langchain.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from google.api_core.retry import Retry
import streamlit as st
from langchain.prompts import PromptTemplate
from geopy.exc import GeocoderServiceError
from geopy.geocoders import Nominatim
from streamlit_folium import folium_static
import folium

# Local folder path
FOLDER_PATH = "C:/Users/Dinesh/Downloads/SampleFolder"  # Replace with your local folder path
SERVICE_ACCOUNT_FILE = 'ornate-charter-376407-1c62e20b756a.json' 
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = SERVICE_ACCOUNT_FILE
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, max_tokens=1000)

# Function to fetch PDFs
def fetch_pdfs_from_local(folder_path):
    pdf_files = []
    if not os.path.exists(folder_path):
        st.sidebar.error("Specified folder does not exist.")
        return []

    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "rb") as f:
                pdf_files.append(io.BytesIO(f.read()))

    if not pdf_files:
        st.sidebar.error("No PDF files found in the specified folder.")
        return []

    return pdf_files

# Function to process PDFs
def process_pdfs(pdf_files):
    all_docs = []
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
        docs = text_splitter.split_documents(data)
        all_docs.extend(docs)

        os.unlink(temp_file_path)
    return all_docs

# Embed and store in FAISS
def embed_and_store(docs):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        request_options={'retry': Retry(initial=0.25, maximum=30.0, multiplier=1.3)}
    )
    vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
    return vectorstore

# Geolocation Functionality
location_cache = {}

def get_location_coordinates(location_name):
    if location_name in location_cache:
        return location_cache[location_name]

    geolocator = Nominatim(user_agent="my-geocoding-app/1.0", timeout=10)
    try:
        location = geolocator.geocode(location_name)
        if location:
            coords = [location.longitude, location.latitude]
            location_cache[location_name] = coords
            return coords
    except GeocoderServiceError as e:
        st.error(f"Geocoding service error for '{location_name}': {e}")
    except Exception as e:
        st.error(f"Error retrieving coordinates for '{location_name}': {e}")

    return None

def get_lat_and_long(location):
    coords = get_location_coordinates(location)
    if coords:
        st.success(f"The coordinates of {location} are {coords[1]}, {coords[0]}")
        m = folium.Map(location=[coords[1], coords[0]], zoom_start=12)
        folium.Marker(location=[coords[1], coords[0]], popup=f"{location}\n({coords[1]}, {coords[0]})").add_to(m)
        folium_static(m)
    else:
        st.error(f"Could not retrieve coordinates for {location}.")

# Main Streamlit app
st.title("Multi-Function App: Q&A and Geospatial Queries")

# PDF Processing
if st.sidebar.button("Fetch and Process PDFs"):
    pdf_files = fetch_pdfs_from_local(FOLDER_PATH)
    if pdf_files:
        docs = process_pdfs(pdf_files)
        vectorstore = embed_and_store(docs)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        system_prompt = """
        You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question and you give a detailed lengthy answer. 
        If you don't know the answer, say that you don't know. 
        Keep your answer clear and concise.
        """
        prompt_template = PromptTemplate(
            input_variables=["input", "context"],
            template=system_prompt + "\n\nContext: {context}\n\nQuestion: {input}\nAnswer:"
        )
        question_answer_chain = create_stuff_documents_chain(llm, prompt=prompt_template)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        st.session_state["rag_chain"] = rag_chain
        st.sidebar.success("All PDFs processed and embedded!")

# Geospatial Queries
st.write("### Enter a location name to get its coordinates and map:")
location_query = st.text_input("Location Query:")
if location_query:
    get_lat_and_long(location_query)

# Document-based Q&A
if "rag_chain" in st.session_state:
    st.write("### Ask questions about the documents:")
    doc_query = st.text_input("Your question about documents:")
    if doc_query:
        response = st.session_state["rag_chain"].invoke({"input": doc_query})
        st.write(f"**Bot:** {response['answer']}")