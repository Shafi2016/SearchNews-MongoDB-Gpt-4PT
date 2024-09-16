import streamlit as st
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import pandas as pd
from datetime import datetime, timedelta
from unidecode import unidecode
import hashlib
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random 
import dns.resolver

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union
from langchain.memory import ConversationBufferMemory
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

# Configure DNS resolver
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8']

# Get secrets
MONGODB_URL = st.secrets["general"]["MONGODB_URL"]
openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]

# MongoDB connection function
@st.cache_resource
def init_mongodb_connection():
    try:
        client = MongoClient(MONGODB_URL, 
                             serverSelectionTimeoutMS=5000,
                             connectTimeoutMS=5000,
                             socketTimeoutMS=5000)
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        return client
    except ConnectionFailure:
        st.error("Failed to connect to MongoDB. Please check your connection string and network.")
        return None
    except ServerSelectionTimeoutError as e:
        st.error(f"MongoDB server selection timeout. Error: {e}")
        return None

# Initialize MongoDB connection
client = init_mongodb_connection()

if client is None:
    st.stop()

# Test the connection
try:
    db = client['data']
    collection = db['articles']
    # Perform a simple operation
    doc_count = collection.count_documents({})
    st.success(f"Successfully connected to MongoDB. Found {doc_count} documents in the 'articles' collection.")
except Exception as e:
    st.error(f"Error accessing MongoDB: {e}")
    st.stop()

# Date inputs
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Start date", datetime.now() - timedelta(days=7))
with col2:
    end_date = st.date_input("End date", datetime.now())

# Ensure end date is not before start date
if start_date > end_date:
    st.error("Error: End date must be after start date.")
    st.stop()

# Query input
query = st.text_input("Enter your query")

# Button to generate summary
if st.button("Generate Summary"):
    if not query:
        st.error("Please enter a query.")
    else:
        with st.spinner("Generating summary..."):
            # Function to clean text
            def clean_text(text):
                if pd.isnull(text):
                    return ""
                text = str(text).strip()
                text = unidecode(text).strip()
                return text

            # Function to create a hash of the text
            def text_hash(text):
                return hashlib.md5(text.encode()).hexdigest()

            # Function to find near-duplicate articles
            def find_near_duplicates(df, threshold=0.95):
                tfidf = TfidfVectorizer().fit_transform(df['cleaned_article'])
                cosine_similarities = cosine_similarity(tfidf, tfidf)
                
                near_duplicates = []
                for i in range(len(df)):
                    for j in range(i+1, len(df)):
                        if cosine_similarities[i][j] > threshold:
                            near_duplicates.append((i, j))
                
                return near_duplicates

            # Fetch and preprocess data
            try:
                start_datetime = datetime.combine(start_date, datetime.min.time())
                end_datetime = datetime.combine(end_date, datetime.max.time())
                date_query = {'date': {'$gte': start_datetime, '$lte': end_datetime}}
                data = collection.find(date_query)
                df = pd.DataFrame(list(data))
                if df.empty:
                    st.warning("No articles found for the selected date range.")
                    st.stop()
                df = df[['date', 'link', 'title', 'article']]
            except Exception as e:
                st.error(f"Error fetching data from MongoDB: {e}")
                st.stop()

            # Clean and hash text
            df['cleaned_article'] = df['article'].apply(clean_text)
            df['cleaned_title'] = df['title'].apply(clean_text)
            df['article_hash'] = df['cleaned_article'].apply(text_hash)
            df['title_hash'] = df['cleaned_title'].apply(text_hash)

            # Remove exact duplicates
            df = df.drop_duplicates(subset=['article_hash', 'title_hash'])

            # Remove near-duplicates
            near_duplicates = find_near_duplicates(df)
            indices_to_remove = set()
            for i, j in near_duplicates:
                if len(df.iloc[i]['cleaned_article']) >= len(df.iloc[j]['cleaned_article']):
                    indices_to_remove.add(j)
                else:
                    indices_to_remove.add(i)

            df = df.drop(df.index[list(indices_to_remove)])
            df = df.reset_index(drop=True)  # Reset index after removing duplicates

            st.write(f"Number of articles found: {len(df)}")

            # LangChain setup
            embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key=openai_api_key)

            # Set up GPT-4o-mini for article selection and summary generation
            llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-4o-mini")

            # Article Selection Tool
            def article_selection(query: str) -> List[int]:
                article_selection_prompt = f"""
                Given the following query: "{query}"
                
                Please analyze ALL of the following articles and select the most relevant ones. 
                It's crucial to consider every article provided, as each may contain important information.
                Return the indices of the selected articles as a Python list of integers.
                Select the number of articles based on their relevance to the query. Only select articles that contribute valuable information.
                Do not select duplicate or irrelevant articles.
                
                Articles:
                """
                
                for i, row in df.iterrows():
                    article_selection_prompt += f"{i}: Title: {row['cleaned_title']}\nSnippet: {row['cleaned_article'][:200]}...\n\n"
                
                try:
                    st.write("Thinking...")
                    response = llm.predict(article_selection_prompt)
                    st.write("Received response from GPT-4o")
                    
                    selected_indices = [int(index) for index in re.findall(r'\b\d+\b', response) if int(index) < len(df)]
                    selected_indices = list(dict.fromkeys(selected_indices))  # Remove duplicates while preserving order
                    
                    return selected_indices
                
                except Exception as e:
                    st.write(f"Error in article selection: {str(e)}. Using random articles with unique titles.")
                    selected_indices = random.sample(range(len(df)), min(10, len(df)))  # Default to 10 random articles in case of error
                    return selected_indices

            # Summary Generation Tool
            def summary_generation(query: str, selected_indices: List[int]) -> str:
                selected_df = df.iloc[selected_indices]
                
                documents = [
                    Document(
                        page_content=row['cleaned_article'],
                        metadata={"title": row['cleaned_title'], "link": row['link'], "date": str(row['date']), "index": index}
                    ) for index, row in selected_df.iterrows()
                ]

                vector_store = FAISS.from_documents(documents, embeddings)
                retriever = vector_store.as_retriever(search_kwargs={"k": len(documents)})

                summary_template = """
                        You are an AI assistant tasked with generating a comprehensive summary report based on the given
