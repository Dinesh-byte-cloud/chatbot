import streamlit as st
import sqlite3
from transformers import BertForQuestionAnswering, BertTokenizer
import torch

# Set up the database connection and cursor
conn = sqlite3.connect('bertchatbot.db')
cursor = conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS qa(
                    qid INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT UNIQUE,
                    answer TEXT
                    )''')
conn.commit()

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
text_tokens = None
inputs = {}

def select_File(file_content):
    global text_tokens
    if file_content:
        print("Selected file")
        text = file_content
        text_tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        print(text_tokens['input_ids'].size())

def display_answer(text):
    global inputs
    if text_tokens is not None:
        user_question = text
        if user_question:
            question_tokens = tokenizer(user_question, return_tensors='pt', padding=True, truncation=True)
            inputs = {
                'input_ids': torch.cat([question_tokens['input_ids'], text_tokens['input_ids']], dim=1),
                'attention_mask': torch.cat([question_tokens['attention_mask'], text_tokens['attention_mask']], dim=1)
            }
            if 'input_ids' in inputs and 'attention_mask' in inputs:
                outputs = model(**inputs)
                start_index = torch.argmax(outputs.start_logits)
                end_index = torch.argmax(outputs.end_logits) + 1
                answer_tokens = inputs['input_ids'][0][start_index:end_index]
                answer = tokenizer.decode(answer_tokens)

                try:
                    cursor.execute("SELECT * FROM qa WHERE question = ?", (user_question,))
                    question_exist = cursor.fetchone()
                    if question_exist:
                        cursor.execute("UPDATE qa SET answer = ? WHERE question = ?", (answer, user_question))
                    else:
                        cursor.execute("INSERT INTO qa(question, answer) VALUES (?, ?)", (user_question, answer))
                    conn.commit()
                except sqlite3.Error as err:
                    print(f"Error connecting to database: {err}")
                return answer
            else:
                print("Error: the inputs do not have input_ids")
    return None


st.set_page_config(page_title="🗣️Chatbot")

# Function to check if the provided password is correct
def check_password():
    if st.session_state["password"] == "Dinesh":
        st.session_state["password_correct"] = True
    else:
        st.session_state["password_correct"] = False

# Default state for password check and upload initiation
if "password_correct" not in st.session_state:
    st.session_state["password_correct"] = False
if "upload_initiated" not in st.session_state:
    st.session_state["upload_initiated"] = False
if "processed_answer" not in st.session_state:
    st.session_state["processed_answer"] = ""

# Button to initiate the upload process
if st.button("Initiate Upload"):
    st.session_state["upload_initiated"] = True

# Password input (appears only after initiating upload)
if st.session_state["upload_initiated"]:
    st.text_input("Enter Password", type="password", on_change=check_password, key="password")

# Upload button (enabled only if the correct password is provided)
uploaded_file = None
if st.session_state["password_correct"]:
    uploaded_file = st.file_uploader("Upload File")

# Provide a default file if upload is not initiated
if not st.session_state["upload_initiated"]:
    default_file_path = "C:/Users/Dinesh/Flask_applications/BertChatbotfortext/Sample.txt"
    try:
        with open(default_file_path, "r") as file:
            default_file_content = file.read()
        select_File(default_file_content)
    except FileNotFoundError:
        st.error("Default file not found. Please check the path.")
else:
    if uploaded_file is not None:
        uploaded_file_content = uploaded_file.read().decode("utf-8")
        select_File(uploaded_file_content)

# Text entry box with placeholder
user_question = st.text_input("Type in your question")

# Answer button
if st.button("Answer"):
    if user_question:
        answer = display_answer(user_question)
        if answer:
            st.session_state["processed_answer"] = answer
        else:
            st.session_state["processed_answer"] = "Could not find an answer."
    else:
        st.session_state["processed_answer"] = "Please type in a question."

# Display the processed text input
if st.session_state["processed_answer"]:
    answer = st.write(f"Your Answer: {st.session_state['processed_answer']}")



