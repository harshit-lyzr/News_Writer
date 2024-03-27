import streamlit as st
import os
from dotenv import load_dotenv,find_dotenv
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent,Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
import json
from PIL import Image
import requests
from bs4 import BeautifulSoup
import re
import ssl

load_dotenv(find_dotenv())
serpapi = os.getenv("SERPAPI_API_KEY")
api=os.getenv("OPENAI_API_KEY")
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

st.set_page_config(
    page_title="Lyzr Personal News Agent",
    layout="centered",  # or "wide"
    initial_sidebar_state="auto",
    page_icon="lyzr-logo-cut.png",
)

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Lyzr Personal News Agent")
st.markdown("### Welcome to the Lyzr Personal News Agent!")
open_ai_text_completion_model = OpenAIModel(
    api_key=api,
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)

def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q": query,
        "gl": "in",
    })

    headers = {
        'X-API-KEY': serpapi,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    res = response.json()
    # Print the response JSON object to inspect its structure

    mys = []
    for item in res.get('organic', []):
        mys.append(item.get('link'))
    return mys

def extract_text_from_url(url):
    try:
        # Fetch HTML content from the URL
        response = requests.get(url)
        response.raise_for_status()

        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract text content and replace consecutive spaces with a maximum of three spaces
        text_content = re.sub(r'\s{4,}', '   ', soup.get_text())

        return text_content

    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from {url}: {e}")
        return None

def extracteddata(query):
    result =  search(query)
    my_data = []
    for i in result:
        get_data = extract_text_from_url(i)
        my_data.append(get_data)
    return my_data[:6]

give_news=st.text_input("search")


if st.button("Get News"):
    data=extracteddata(give_news)

    writer_agent = Agent(
            role="writer",
            prompt_persona=f"""You are an expert News Writer and You have to write news from {data} in 300 words include all hot topics from {data}
            """
        )

    task1  =  Task(
        name="news writing",
        model=open_ai_text_completion_model,
        agent=writer_agent,
        instructions=f'Analyse {data} and write news article from {data} in 250 words',
    )
    output = LinearSyncPipeline(
        name="Game Pipline",
        completion_message="pipeline completed",
        tasks=[
              task1,
        ],
    ).run()

    st.markdown(output[0]['task_output'])
