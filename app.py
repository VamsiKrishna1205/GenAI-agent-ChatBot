import streamlit as st
import pandas as pd
import faiss
import pickle
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Agent Connect Bot",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Agent Connect Bot")
st.write("Find the best real estate agent using AI")

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("real_estate_agents_dataset_1000_rows.csv")

df["text"] = df.apply(
    lambda x: f"{x.agent_name} is a real estate agent in {x.city} specializing in {x.specialization}. "
              f"They have {x.experience_years} years of experience and rating {x.rating}.",
    axis=1
)

documents = df["text"].tolist()

# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# CREATE FAISS INDEX
# -----------------------------
embeddings = embedding_model.encode(documents)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# -----------------------------
# GEMINI CONFIG
# -----------------------------
genai.configure(api_key="AIzaSyDzQu10PoC0cyRmeXxkRWRX99jU5Rg33vU")

llm = genai.GenerativeModel("gemini-2.5-flash")

# -----------------------------
# RETRIEVAL FUNCTION
# -----------------------------
def retrieve_agents(query, k=5):

    query_vector = embedding_model.encode([query])

    distances, indices = index.search(query_vector, k)

    results = [documents[i] for i in indices[0]]

    return results

# -----------------------------
# RAG RESPONSE
# -----------------------------
def agent_connect_bot(query):

    retrieved_docs = retrieve_agents(query)

    context = "\n".join(retrieved_docs)

    prompt = f"""
You are an assistant that connects buyers with real estate agents.

Use the following agent information to recommend agents.

Agent Data:
{context}

User Question:
{query}

Provide the best agent recommendation.
"""

    response = llm.generate_content(prompt)

    return response.text

# -----------------------------
# CHAT MEMORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# -----------------------------
# USER INPUT
# -----------------------------
user_input = st.chat_input("Ask about real estate agents...")

if user_input:

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    # Generate response
    response = agent_connect_bot(user_input)

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )