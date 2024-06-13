import os
import streamlit as st
import openai
from elasticsearch import Elasticsearch

# This code is part of an Elastic Blog showing how to combine
# Elasticsearch's search relevancy power with 
# OpenAI's GPT's Question Answering power
# https://www.elastic.co/blog/chatgpt-elasticsearch-openai-meets-private-data

# Code is presented for demo purposes but should not be used in production
# You may encounter exceptions which are not handled in the code


# Required Environment Variables
# openai_api - OpenAI API Key
# cloud_id - Elastic Cloud Deployment ID
# cloud_user - Elasticsearch Cluster User
# cloud_pass - Elasticsearch User Password

openai.api_key = os.environ['AZURE_OPENAI_KEY']
openai.api_base = os.environ['AZURE_OPENAI_ENDPOINT']
openai.api_version = '2023-05-15'
openai.api_type = "azure"
model = "gpt-3.5-turbo-0301"

# Connect to Elastic Cloud cluster
def es_connect(cid, user, passwd):
    es = Elasticsearch(cloud_id=cid, http_auth=(user, passwd))
    return es

# Search ElasticSearch index and return body and URL of the result
def search(query_text):
    cid = os.environ['cloud_id']
    cp = os.environ['cloud_pass']
    cu = os.environ['cloud_user']
    es = es_connect(cid, cu, cp)

    # Elasticsearch query (BM25) and kNN configuration for hybrid search
    query = {
        "bool": {
            "must": [{
                "match": {
                    "title": {
                        "query": query_text,
                        "boost": 1
                    }
                }
            }],
            "filter": [{
                "exists": {
                    "field": "bndes2-title-vector"
                }
            }]
        }
    }

    knn = {
        "field": "bndes2-title-vector",
        "k": 1,
        "num_candidates": 20,
        "query_vector_builder": {
            "text_embedding": {
                "model_id": "sentence-transformers__all-distilroberta-v1",
                "model_text": query_text
            }
        },
        "boost": 24
    }

    fields = ["title", "body_content", "url"]
    index = 'search-bndes2'
    resp = es.search(index=index,
                     query=query,
                     knn=knn,
                     fields=fields,
                     size=1,
                     source=False)

    body = resp['hits']['hits'][0]['fields']['body_content'][0]
    url = resp['hits']['hits'][0]['fields']['url'][0]

    return body, url

def truncate_text(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text

    return ' '.join(tokens[:max_tokens])

# Generate a response from ChatGPT based on the given prompt
def chat_gpt(prompt, model="gpt-3.5-turbo", max_tokens=1024, max_context_tokens=4000, safety_margin=15):
    # Truncate the prompt content to fit within the model's context length
    truncated_prompt = truncate_text(prompt, max_context_tokens - max_tokens - safety_margin)

    response = openai.ChatCompletion.create(model=model, deployment_id="cosmo-sa",
                                            messages=[{"role": "system", "content": "Você é um assistente bastante esforçado em ajudar."}, {"role": "user", "content": truncated_prompt}])

    return response["choices"][0]["message"]["content"]


st.title("BNDES GPT")

# Main chat form
with st.form("chat_form"):
    query = st.text_input("Você: ")
    submit_button = st.form_submit_button("Enviar")

# Generate and display response on form submission
negResponse = "Não foi possível encontrar a resposta com base nas informações que foram ingeridas do portal do BNDES até o momento."
if submit_button:
    resp, url = search(query)
    prompt = f"Responda essa questão: {query}\nUsando somente a informação deste documento: {resp}\nCaso a resposta não esteja na documentação fornecida, responda '{negResponse}' e nada mais"
    answer = chat_gpt(prompt)
    
    if negResponse in answer:
        st.write(f"BNDESGPT: {answer.strip()}")
    else:
        st.write(f"BNDESGPT: {answer.strip()}\n\nDocs: {url}")
