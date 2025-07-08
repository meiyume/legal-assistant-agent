import streamlit as st
from openai import OpenAI
from langchain.llms import OpenAI as LangOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain, LLMChain
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
acting_as = st.radio("Are you writing this letter in your own name or on behalf of a client?", ["Myself", "On behalf of a client"])
user_name = st.text_input("Your Full Name")
user_address = st.text_input("Your Address")

if acting_as == "On behalf of a client":
    client_name = st.text_input("Client's Full Name")
    client_address = st.text_input("Client's Address")
else:
    client_name = user_name
    client_address = user_address

opponent_name = st.text_input("Name of other party (Landlord/Employer/etc)")
opponent_address = st.text_input("Address of other party")
description = st.text_area("Briefly describe the issue:")
event_date = st.date_input("When did the issue occur?", datetime.date.today())
letter_type = st.radio("Generate: ", ["Letter of Demand", "Complaint Letter"])

# Step 3: Generate Letter
if st.button("Generate Letter"):
    if not (user_name and opponent_name and description and client_name and client_address):
        st.error("Please complete all required fields.")
    else:
        # Build dynamic parts of the letter
        if acting_as == "Myself":
            sender_identity = f"{client_name}\n{client_address}"
            intro_line = f"I am writing regarding the following matter."
        else:
            sender_identity = f"{client_name}\n{client_address}\n(c/o {user_name}, {user_address})"
            intro_line = f"I am writing on behalf of my client, {client_name}, regarding the following matter."

        # Chain 1: Summarize the issue
        summary_prompt = PromptTemplate(
            input_variables=["description"],
            template="""
Summarize the legal issue in 1-2 sentences clearly and formally.

Description: {description}
            """
        )
        summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")

        # Chain 2: Generate the letter
        letter_prompt = PromptTemplate(
            input_variables=[
                "summary", "sender_identity", "opponent_name", "opponent_address",
                "letter_type", "topic", "event_date", "intro_line"
            ],
            template="""
You are a legal assistant.

Draft a formal {letter_type} letter using the following details:

From:
{sender_identity}

To:
{opponent_name}
{opponent_address}

Topic: {topic}
Date of Issue: {event_date}

Opening:
{intro_line}

Summary of the issue:
{summary}

Write a professional letter including:
- Date and subject
- Salutation
- Body with background, demands, and any deadlines
- Closing and polite sign-off
            """
        )
        letter_chain = LLMChain(llm=llm, prompt=letter_prompt, output_key="letter")

        # Combine both chains
        overall_chain = SequentialChain(
            chains=[summary_chain, letter_chain],
            input_variables=[
                "description", "sender_identity", "opponent_name", "opponent_address",
                "letter_type", "topic", "event_date", "intro_line"
            ],
            output_variables=["summary", "letter"]
        )

        # Run the chain
        result = overall_chain({
            "description": description,
            "sender_identity": sender_identity,
            "opponent_name": opponent_name,
            "opponent_address": opponent_address,
            "letter_type": letter_type,
            "topic": topic,
            "event_date": str(event_date),
            "intro_line": intro_line
        })

        # Display output
        st.subheader("📄 Summary of Issue")
        st.write(result["summary"])

        st.subheader("📄 Generated Letter")
        st.code(result["letter"], language="markdown")
        st.download_button("Download as .txt", result["letter"], file_name="generated_letter.txt")

