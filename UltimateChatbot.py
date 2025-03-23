import streamlit as st
import os
import google.generativeai as genai
from docx import Document

st.set_page_config(page_title="💬Textbot")

# Sidebar for configuration and API key input
with st.sidebar:
    st.title('💬Textbot')

    # Check if API token is already provided
    if 'GOOGLE_API_TOKEN' in st.secrets:
        st.success('API key already provided!', icon='✅')
        gemini_ai = st.secrets['GOOGLE_API_TOKEN']
    else:
        # Prompt for Google Gen AI token
        gemini_ai = st.text_input('Enter Gen AI credentials:', type='password')
        if not gemini_ai.startswith('AIz'):
            st.warning('Please enter your credentials!', icon='⚠️')
        else:
            st.success('Proceed to entering your prompt message!', icon='👉')

    # Set environment variable for the token
    os.environ['gemini_ai_TOKEN'] = gemini_ai

    st.subheader('Models and parameters')

    # Configure the Google Gen AI API
    genai.configure(api_key=gemini_ai)

    # Select model (for future expansion if multiple models are needed)
    model = genai.GenerativeModel('gemini-pro')
    selected_model = st.selectbox('Model', ['Small language model'], key='selected_model')

# Store LLM generated responses and track if last response is from API
if "messages" not in st.session_state.keys():
    st.session_state.messages = []
    st.session_state.is_last_api_response = False



def read_wordFile(file_path):
    doc = Document(file_path)
    data=[]
    for x in doc.paragraphs:
        data.append(x.text)
    return "\n".join(data)

# Dropdown logic
def handle_dropdown():
    dropdown = st.selectbox("How may I help you?", ["Please select one query from below", "Summarize this text file"])

    if dropdown == "Summarize this text file":
       uploaded_file = st.file_uploader("Choose a file")
       Word_data = read_wordFile(uploaded_file)
       st.session_state.dropdown_response = Word_data
    else:
        st.session_state.dropdown_response = None

       

# Display chat messages
def display_messages():
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.write(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant"):
                st.write(message["content"])
        elif message["role"] == "api":
            with st.chat_message("api"):
                st.write(message["content"])
        else:
            continue


# Generate response using Google Gen AI
def generate_google_genai_response(prompt_input, exclude_file_content=True):
    # Gather all user inputs for context
    string_dialogue = ""
    for dict_message in st.session_state.messages:
        if exclude_file_content and dict_message["role"] == "api":
            continue  # Skip file content for follow-up prompts
        if dict_message["role"] == "user":
            string_dialogue += "User: " + str(dict_message["content"]) + "\n\n"
        else:
            string_dialogue += "Assistant: " + str(dict_message["content"]) + "\n\n"
    
    # Generate response using Google Gen AI model
    response = model.generate_content(f"{string_dialogue} {prompt_input} Assistant: Write the answer within 100 words and only answer from the file.")
    
    # Extract text content from the response
    return response.text  # Access the first generated content


# Handle dropdown for API response
handle_dropdown()
display_messages()

if prompt := st.chat_input(disabled=not gemini_ai):
    st.session_state.dropdown_response = None
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.is_last_api_response = True
    with st.chat_message("user"):
        st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_google_genai_response(prompt, exclude_file_content=True)
                placeholder = st.empty()
                placeholder.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})



# Handle submission of dropdown response
if "dropdown_response" in st.session_state and st.session_state.dropdown_response:
    api_response = st.session_state.dropdown_response
    if not st.session_state.is_last_api_response:
        st.session_state.messages.append({"role": "api", "content": api_response})
        st.session_state.is_last_api_response = True

        # Generate a new response if last message is not from assistant
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = generate_google_genai_response(api_response)
                    placeholder = st.empty()
                    placeholder.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.dropdown_response = None

