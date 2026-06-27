import json
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, AnnSearchRequest, Function, FunctionType
from langchain_milvus import Milvus
import os


def emb_text(text, embedding_model):
    emb = embedding_model.encode(text,normalize_embeddings=True)
    return emb.tolist()


def embedding_search(milvus_client, question, collection_name, embedding_model, emb_field, metric_type, output_fields):
    
    emb_results = milvus_client.search(
        collection_name=collection_name,
        data=[emb_text(question, embedding_model)],
        anns_field=emb_field,
        limit=3,
        search_params={"metric_type": metric_type, "params": {}},
        output_fields=output_fields,
    )
    
    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in emb_results[0]
    ]
    
    return retrieved_lines_with_distances



def bm25_search(milvus_client, collection_name, question, sparse_field, output_fields):
    search_params = {}
    
    bm25_results = milvus_client.search(
        collection_name=collection_name,
        data=[question],
        anns_field=sparse_field,
        output_fields=output_fields,
        limit=3,
        search_params=search_params
    )
    
    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in bm25_results[0]
    ]
    
    return retrieved_lines_with_distances



def hybrid_search(milvus_client, collection_name, question, embedding_model, emb_field, sparse_field, emb_metric, ranker, output_fields):
    dense_req = AnnSearchRequest(
        data=[emb_text(question, embedding_model)],
        anns_field=emb_field,
        param={"metric_type": emb_metric, "params": {}},
        limit=10
    )

    bm25_req = AnnSearchRequest(
        data=[question],
        anns_field=sparse_field,
        param={"metric_type": "BM25", "params": {}},
        limit=10
    )
    
    hybrid_results = milvus_client.hybrid_search(
        collection_name,
        [dense_req, bm25_req],
        ranker=ranker,
        limit=10,
        output_fields=output_fields
    )
    
    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in hybrid_results[0]
    ]
    
    return retrieved_lines_with_distances