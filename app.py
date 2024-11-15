from flask import Flask, render_template, request, jsonify 
import openai 
  
"""from haystack.telemetry import tutorial_running"""
import logging
"""from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import fetch_archive_from_http, convert_files_to_docs, clean_wiki_text
from haystack.nodes import BM25Retriever, EmbeddingRetriever, FARMReader
from haystack.nodes import BM25Retriever, EmbeddingRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers
from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents
from haystack.pipelines import Pipeline
from haystack.nodes import JoinDocuments"""

from haystack import Pipeline
from haystack import BaseComponent
from typing import Optional, List

import json
import pprint
import sys

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING,filename='example.log', encoding='utf-8')
#logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)
logger = logging.getLogger(__name__)

document_store = InMemoryDocumentStore(use_bm25=True)

# Download and prepare data 
doc_dir = "data/articles"

got_docs = convert_files_to_docs(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
document_store.delete_documents()
document_store.write_documents(got_docs)

bm25_retriever = BM25Retriever()

from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents

p_retrieval = DocumentSearchPipeline(bm25_retriever)

# Initialize Sparse Retriever
bm25_retriever = BM25Retriever(document_store=document_store)

# Initialize embedding Retriever
embedding_retriever = EmbeddingRetriever(
    document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)
document_store.update_embeddings(embedding_retriever, update_existing_embeddings=False)

# Initialize Reader
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")

# Create ensembled pipeline
p_ensemble = Pipeline()
p_ensemble.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
p_ensemble.add_node(component=embedding_retriever, name="EmbeddingRetriever", inputs=["Query"])
p_ensemble.add_node(
    component=JoinDocuments(join_mode="concatenate"), name="JoinResults", inputs=["BM25Retriever", "EmbeddingRetriever"]
)
p_ensemble.add_node(component=reader, name="Reader", inputs=["JoinResults"])

#res = p_ensemble.run(
#    query=str, params={"EmbeddingRetriever": {"top_k": 5}, "BM25Retriever": {"top_k": 5}}
#)
  
app = Flask(__name__) 
  
# OpenAI API Key 
openai.api_key = 'HuggingFaceH4/zephyr-7b-beta'

f = open("demofile.txt", "a")

  
def get_completion(prompt): 
    print(prompt) 
    logger.debug("Prompt:")
    logger.debug(prompt)

    query2 = openai.Completion.create( 
        engine="text-davinci-003", 
        prompt=prompt, 
        max_tokens=1024, 
        n=1, 
        stop=None, 
        temperature=0.5, 
    ) 

    p_ensemble = Pipeline()
    p_ensemble.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
    p_ensemble.add_node(component=embedding_retriever, name="EmbeddingRetriever", inputs=["Query"])
    p_ensemble.add_node(
    component=JoinDocuments(join_mode="concatenate"), name="JoinResults", inputs=["BM25Retriever", "EmbeddingRetriever"]
    )
    p_ensemble.add_node(component=reader, name="Reader", inputs=["JoinResults"])
  
    response = query2.choices[0].text 
    str = jsonify({'str': response}) 
    print(response)
    logger.debug(response)
    
    res = p_ensemble.run(
    query=str, params={"EmbeddingRetriever": {"top_k": 5}, "BM25Retriever": {"top_k": 5}}
)
    result = pprint.pformat(res)
    f.write(result)
    #return response 
    return result
  
@app.route("/", methods=['POST', 'GET']) 
def query_view(): 
    if request.method == 'POST': 
        print('step1') 
        prompt = request.form['prompt'] 
        response = get_completion(prompt) 
        print(response) 
        logger.debug(response)
        f.write(response)
        return jsonify({'response': response}) 
    return render_template('index.html') 

f.close()
  
if __name__ == "__main__": 
    app.run(debug=True) 
    