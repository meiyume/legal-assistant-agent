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

# Step 3: Letter Style
letter_type = st.radio("Generate: ", ["Letter of Demand", "Complaint Letter"])

# Step 4: Generate Letter
if st.button("Generate Letter"):
    if not (user_name and opponent_name and description and client_name and client_address):
        st.error("Please complete all required fields.")
    else:
        # Chain 1: Summarize issue
        summary_prompt = PromptTemplate(
            input_variables=["description"],
            template="""
Summarize the legal issue in 2-3 sentences clearly and formally:

Description: {description}
            """
        )
        summary_chain = LLMChain(llm=llm, prompt=summary_prompt, output_key="summary")

        # Chain 2: Generate letter based on summary
        letter_prompt = PromptTemplate(
            input_variables=[
                "summary", "client_name", "client_address", "user_name", "user_address",
                "opponent_name", "opponent_address", "letter_type", "topic", "event_date", "acting_as"
            ],
            template="""
You are a legal assistant.
Draft a {letter_type} based on the following:
- Topic: {topic}
- Date of Issue: {event_date}
- From: {client_name}
- Client Address: {client_address}
- Prepared by: {user_name} ({user_address})
- To: {opponent_name}
- Opponent Address: {opponent_address}
- Summary of Issue: {summary}

If 'acting_as' is 'Myself', the letter should be written in first person's perspective.
If 'acting_as' is 'On behalf of a client', the letter should be written in third person's perspective.

Write this as a formal letter with:
- Date and subject
- Salutation
- Body with context, action demanded, and timeframe
- Closing statement and sign-off
            """
        )
        letter_chain = LLMChain(llm=llm, prompt=letter_prompt, output_key="letter")

        # Sequential chain
        overall_chain = SequentialChain(
            chains=[summary_chain, letter_chain],
            input_variables=[
                "description", "client_name", "client_address", "user_name", "user_address",
                "opponent_name", "opponent_address", "letter_type", "topic", "event_date", "acting_as"
            ],
            output_variables=["summary", "letter"]
        )

        # Run and capture output as dict
        result = overall_chain({
            "description": description,
            "client_name": client_name,
            "client_address": client_address,
            "user_name": user_name,
            "user_address": user_address,
            "opponent_name": opponent_name,
            "opponent_address": opponent_address,
            "letter_type": letter_type,
            "topic": topic,
            "event_date": str(event_date),
            "acting_as": acting_as
        })

        # Display results
        st.subheader("📄 Summary of Issue")
        st.write(result["summary"])

        st.subheader("📄 Generated Letter")
        st.code(result["letter"], language='markdown')
        st.download_button("Download as .txt", result["letter"], file_name="generated_letter.txt")
