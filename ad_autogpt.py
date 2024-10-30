# -*- coding: utf-8 -*-
"""AD-AUTOGPT.ipynb

"""

import os
import google.generativeai as genai
import spacy
import requests
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from datetime import datetime
from bs4 import BeautifulSoup

# Set up Gemini API key
genai.configure(api_key='YOUR_GEMINI_API_KEY')
model = genai.GenerativeModel("gemini-1.5-pro")

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Fetch articles from a single open-source site: NIH on Alzheimer's Disease
def retrieve_alzheimers_data():
    url = 'https://www.nia.nih.gov/news/topics/alzheimers'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    response = requests.get(url, headers=headers)
    articles = []
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = soup.find_all('div', class_='views-row')
        
        for item in news_items:
            # Extract metadata: title, date, and link
            title = item.find('h3').get_text(strip=True)
            date_text = item.find('span', class_='date-display-single').get_text(strip=True)
            date_published = datetime.strptime(date_text, '%B %d, %Y')
            link = 'https://www.nia.nih.gov' + item.find('a')['href']
            
            # Follow the link to get the actual article content
            article_response = requests.get(link, headers=headers)
            
            if article_response.status_code == 200:
                article_soup = BeautifulSoup(article_response.text, 'html.parser')
                
                # Find the div with class 'content content-inner content_full col-12-full'
                content_div = article_soup.find('div', class_='content content-inner content_full col-12-full')
                
                # Check if the content div is found and extract its text content
                article_text = content_div.get_text(strip=True) if content_div else "Content not found"
                
                # Append article metadata along with content
                articles.append({
                    'title': title,
                    'link': link,
                    'date': date_published,
                    'content': article_text  # Full content of the article
                })
    
    return articles

def summarize_content(text):
    response = model.predict(f"Summarize this content: {text}")
    return response.text.strip()

def extract_locations(text):
    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "GPE"]

def perform_topic_modeling(docs, num_topics=5):
    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(docs)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    lda.fit(doc_term_matrix)
    terms = vectorizer.get_feature_names_out()
    topics = [[terms[i] for i in topic.argsort()[-5:]] for topic in lda.components_]
    return topics

def plot_locations(locations):
    unique_locations = set(locations)
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(unique_locations)), [1] * len(unique_locations), marker='o')
    plt.xticks(range(len(unique_locations)), list(unique_locations), rotation=45)
    plt.title("Geographical Mentions in Alzheimer's Data")
    plt.show()

def validate_output(instruction, output):
    validation_prompt = (
        f"Instruction: '{instruction}'\n"
        f"Output: '{output}'\n"
        f"Is this output valid for the given instruction? (Answer 'yes' or 'no')"
    )
    response = model.predict(validation_prompt)
    return response.text.strip().lower() == 'yes'

tools = [
    Tool(name="RetrieveArticle", func=retrieve_alzheimers_data, description="Fetches Alzheimer's-related articles"),
    Tool(name="SummarizeContent", func=summarize_content, description="Summarizes article content"),
    Tool(name="ExtractLocations", func=extract_locations, description="Extracts location information from text"),
    Tool(name="TopicModeling", func=perform_topic_modeling, description="Performs topic modeling on a collection of documents"),
    Tool(name="VisualizeLocations", func=plot_locations, description="Plots geographical data of extracted locations")
]

# Create prompt templates for LangChain
decompose_prompt_template = PromptTemplate(
    input_variables=["main_prompt"],
    template="Break the following instruction into smaller actionable sub-prompts: {main_prompt}"
)

instruction_prompt_template = PromptTemplate(
    input_variables=["sub_prompt"],
    template="Identify instruction for: {sub_prompt}"
)

# Define the main AD-AutoGPT function with LangChain
def main_ad_autogpt(main_prompt):
    decompose_prompt_template = PromptTemplate(
        input_variables=["main_prompt"],
        template="Break the following instruction into smaller actionable sub-prompts: {main_prompt}"
    )

    # Create the decomposition chain using LLMChain
    decomposition_chain = LLMChain(llm=model, prompt=decompose_prompt_template)

    # Run the decomposition chain
    sub_prompts = decomposition_chain.run(main_prompt).splitlines()

    # Process each sub-prompt with validation and retry
    results = []
    max_retries = 3
    for sub_prompt in sub_prompts:
        retries = 0
        valid_output = False
        output = None
        
        # Determine the instruction for the sub-prompt using LangChain
        instruction_chain = LLMChain(llm=model, prompt=instruction_prompt_template)
        instruction = instruction_chain.run(sub_prompt).strip()

        # Repeat process if validation fails
        while retries < max_retries and not valid_output:
            # Execute the action based on the instruction
            if "Retrieve" in instruction:
                output = retrieve_alzheimers_data()
            elif "Summarize" in instruction:
                output = summarize_content(output)  # Pass previous data to be summarized
            elif "Extract Locations" in instruction:
                output = extract_locations(output)  # Pass summarized content to extract locations
            elif "Topic Modeling" in instruction:
                output = perform_topic_modeling([output])  # Pass summary list for topic modeling
            elif "Visualize" in instruction:
                plot_locations(output)  # Pass location data to visualize

            # Validate output with Gemini
            valid_output = validate_output(instruction, output)
            if not valid_output:
                print(f"Validation failed for instruction '{instruction}'. Retrying ({retries + 1}/{max_retries})...")
                retries += 1

        # Stop process if validation fails after retries
        if not valid_output:
            print(f"Final validation failed for instruction '{instruction}' after {max_retries} retries. Skipping.")
            continue

        # Store the result and continue if validation passed
        results.append(output)

    return results

# Example usage
if __name__ == "__main__":
    main_prompt = "Analyze recent news on Alzheimer's Disease"
    results = main_ad_autogpt(main_prompt)

    if results:
        print("\nProcessed Results:")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result}")
