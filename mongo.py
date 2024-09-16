
import streamlit as st
from pymongo import MongoClient
import pandas as pd
from datetime import datetime, timedelta
from unidecode import unidecode
import hashlib
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random 

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



MONGODB_URL = st.secrets["general"]["MONGODB_URL"]
openai_api_key = st.secrets["general"]["OPENAI_API_KEY"]

# Establish connection
try:
    client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000)
    client.server_info()  # Forces a call to the server
    st.success("Connected successfully to MongoDB.")
except Exception as e:
    st.error(f"Failed to connect to MongoDB: {e}")

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
            # Establish connection to MongoDB
            try:
                client = MongoClient(MONGODB_URL)
            except Exception as e:
                st.error(f"Failed to connect to MongoDB: {e}")
                st.stop()

            db = client['data']
            collection = db['articles']

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
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.max.time())
            date_query = {'date': {'$gte': start_datetime, '$lte': end_datetime}}
            data = collection.find(date_query)
            df = pd.DataFrame(list(data))
            df = df[['date', 'link', 'title', 'article']]

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
            embeddings = OpenAIEmbeddings(model='text-embedding-3-small', openai_api_key= openai_api_key)

            # Set up GPT-4o-mini for article selection and summary generation
            llm = ChatOpenAI(temperature=0, openai_api_key= openai_api_key, model_name="gpt-4o-mini")

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
                        You are an AI assistant tasked with generating a comprehensive summary report based on the given query and ALL retrieved articles. Your report should be in Markdown format and include:

                        # Summary: {question}

                        ## Introduction
                        Provide a brief overview of the topic and its significance.

                        ## Main Points
                        Summarize the key findings, trends, or arguments related to the query, ensuring you draw information from ALL provided articles. Use subheadings (###) for each main point.

                        ## Conclusion
                        Offer a concise wrap-up of the main ideas and their implications.

                        Use in-text citations to reference specific information from the articles. Citations should be in the format [1], [2], etc.

                        Query: {question}

                        Relevant articles:
                        {context}

                        Please provide a comprehensive summary report based on ALL the above information. It's crucial to consider and incorporate insights from every provided article, as each may contain unique and important information. Ensure that you use in-text citations throughout the report to reference each article used.

                        Use the following format for citations:

                        In-text citation: [1], [2], [3], [4], [5], [6], [7], [8], [9], [10] etc.

                        Use only the information provided in the context. Do not include any information that is not from these articles. 
                        """

                PROMPT = PromptTemplate(
                    template=summary_template, input_variables=["question", "context"]
                )

                chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    return_source_documents=True,
                    combine_docs_chain_kwargs={"prompt": PROMPT}
                )

                result = chain({"question": query, "chat_history": []})
                
                summary = result['answer']
                source_docs = result['source_documents']
                
                # Remove the separate Citations section if it exists
                summary = re.sub(r'\n#+\s*Citations\s*\n+.*?(?=\n#+|\Z)', '', summary, flags=re.DOTALL)
                
                # Remove the Explanation of Selection section if it exists
                summary = re.sub(r'\n#+\s*Explanation of Selection\s*\n+.*?(?=\n#+|\Z)', '', summary, flags=re.DOTALL)
                
                # Generate references
                references = "\n\n## References\n"
                used_docs = []
                
                for i, doc in enumerate(source_docs, 1):
                    if f'[{i}]' in summary:
                        used_docs.append(doc)
                        references += f"{i}. [{doc.metadata['title']}]({doc.metadata['link']}) - {doc.metadata['date']}\n"
                
                # Update citations in the summary
                for i, doc in enumerate(used_docs, 1):
                    summary = summary.replace(f'[{i}]', f'[{i}]')
                
                final_report = summary.strip() + references
                
                return final_report

            # Create tools

            # Global variable to store selected indices
            selected_indices = []

            # Modified Article Selection Tool
            def article_selection_wrapper(query: str) -> str:
                global selected_indices
                selected_indices = article_selection(query)
                return f"Selected {len(selected_indices)} articles for the query: {query}"

            # Modified Summary Generation Tool
            def summary_generation_wrapper(query: str) -> str:
                global selected_indices
                if not selected_indices:
                    return "Please select articles first using the Article Selection tool."
                return summary_generation(query, selected_indices)

            # Create tools using the wrapper functions
            tools = [
                Tool(
                    name="Article Selection",
                    func=article_selection_wrapper,
                    description="Use this tool to select relevant articles based on a query."
                ),
                Tool(
                    name="Summary Generation",
                    func=summary_generation_wrapper,
                    description="Use this tool to generate a summary report based on previously selected articles and a query."
                )
            ]

            # Create prompt
            prompt = ZeroShotAgent.create_prompt(
                tools,
                prefix="""You are an AI assistant tasked with generating a summary report based on news articles. Follow these steps:
                1. Use the Article Selection tool to select relevant articles for the given query.
                2. Then use the Summary Generation tool to create a comprehensive summary based on the selected articles.
                Use the following tools:""",
                suffix="""Begin!

                Question: {input}
                {agent_scratchpad}""",
                input_variables=["input", "agent_scratchpad"]
            )

            # Create LLMChain
            llm_chain = LLMChain(llm=llm, prompt=prompt)

            # Create the agent
            agent = ZeroShotAgent(
                llm_chain=llm_chain,
                tools=tools,
                verbose=True,  return_intermediate_steps=True,
            )

            # Create agent executor
            agent_executor = AgentExecutor.from_agent_and_tools(
                agent=agent, 
                tools=tools, 
                verbose=True,
                return_intermediate_steps=True
            )

            # Run the agent and capture the full output
            result = agent_executor({"input": query})

            # Function to format the output
            def format_output(result):
                output = ""
                for step in result["intermediate_steps"]:
                    action = step[0]
                    observation = step[1]
                    output += f"Action: {action.tool}\n"
                    output += f"Action Input: {action.tool_input}\n"
                    output += f"Observation: {observation}\n\n"
                return output

            # Display the full output (intermediate steps)
            full_output = format_output(result)
            st.text_area("Execution Process", full_output, height=500)

            # Display the final summary report
            st.markdown("## Analysis")
            st.markdown(result["output"])

            # Add a footer
            st.markdown("---")
            st.markdown("Created with Streamlit and LangChain")
