from haystack.telemetry import tutorial_running
import logging
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import fetch_archive_from_http, convert_files_to_docs, clean_wiki_text
from haystack.nodes import BM25Retriever, EmbeddingRetriever, FARMReader
from haystack.nodes import BM25Retriever, EmbeddingRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers
from haystack.pipelines import DocumentSearchPipeline
from haystack.utils import print_documents
from haystack.pipelines import Pipeline
from haystack.nodes import JoinDocuments

from haystack import BaseComponent
from typing import Optional, List

import json
import pprint
import sys

inp = input("Ask a question please.\n")
 
# prints inp
print(inp)

#print(type(inp))
str = '"' + inp + '"' 

#str = "Tell me about leadersship?"

tutorial_running(11)

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

document_store = InMemoryDocumentStore(use_bm25=True)

# Download and prepare data - 517 Wikipedia articles for Game of Thrones
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
"""
p_retrieval = DocumentSearchPipeline(embedding_retriever)
res = p_retrieval.run(query=str, params={"Retriever": {"top_k": 10}})
print_documents(res, max_text_len=200)

# Custom built extractive QA pipeline
p_extractive = Pipeline()
p_extractive.add_node(component=bm25_retriever, name="Retriever", inputs=["Query"])
p_extractive.add_node(component=reader, name="Reader", inputs=["Retriever"])


# Now we can run it
res = p_extractive.run(query=str, params={"Retriever": {"top_k": 10}})
print_answers(res, details="minimum")

"""
# Create ensembled pipeline
p_ensemble = Pipeline()
p_ensemble.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
p_ensemble.add_node(component=embedding_retriever, name="EmbeddingRetriever", inputs=["Query"])
p_ensemble.add_node(
    component=JoinDocuments(join_mode="concatenate"), name="JoinResults", inputs=["BM25Retriever", "EmbeddingRetriever"]
)
p_ensemble.add_node(component=reader, name="Reader", inputs=["JoinResults"])

res = p_ensemble.run(
    query=str, params={"EmbeddingRetriever": {"top_k": 5}, "BM25Retriever": {"top_k": 5}}
)

print_answers(res, details="minimum")

result = pprint.pformat(res)

f = open("demofile4.txt", "a")
f.write(result)
f.close()