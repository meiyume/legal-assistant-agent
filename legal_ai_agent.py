import streamlit as st
from openai import OpenAI
from langchain.llms import OpenAI as LangOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import datetime

# Load API key securely
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
llm = LangOpenAI(openai_api_key=st.secrets["OPENAI_API_KEY"], temperature=0.4)

# Page Config
st.set_page_config(page_title="AI Legal Triage Agent", layout="wide")
st.title("🧑‍⚖️ AI Legal Triage Agent – Initial Legal Help")
st.caption("⚠️ This tool is for general information and letter drafting only. Not legal advice. 😊")

# Step 1: Select Legal Topic
topic = st.selectbox("Select your legal issue:", ["Tenancy Dispute", "Employment Issue", "Contract Breach", "Others"])

# Step 2: User Inputs
user_name = st.text_input("Your Name")
opponent_name = st.text_input("Name of other party (Landlord/Employer/etc)")
description = st.text_area("Briefly describe the issue:")
event_date = st.date_input("When did the issue occur?", datetime.date.today())

# Step 3: Letter Style
letter_type = st.radio("Generate: ", ["Letter of Demand", "Complaint Letter"])

# Step 4: Generate Letter
if st.button("Generate Letter"):
    if not (user_name and opponent_name and description):
        st.error("Please complete all required fields.")
    else:
        template = PromptTemplate(
            input_variables=["user_name", "opponent_name", "description", "letter_type", "topic", "event_date"],
            template="""
You are a legal assistant.
Write a {letter_type} based on the following:
- Topic: {topic}
- Date of Issue: {event_date}
- From: {user_name}  
- To: {opponent_name}  
- Description: {description}

Structure it formally with date, subject, salutation, body, and closing.
        """
        )

        chain = LLMChain(llm=llm, prompt=template)
        response = chain.run({
            "user_name": user_name,
            "opponent_name": opponent_name,
            "description": description,
            "letter_type": letter_type,
            "topic": topic,
            "event_date": str(event_date)
        })

        st.subheader("📄 Generated Letter")
        st.code(response, language='markdown')
        st.download_button("Download as .txt", response, file_name="generated_letter.txt")
