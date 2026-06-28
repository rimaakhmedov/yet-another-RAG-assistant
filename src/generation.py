from huggingface_hub import InferenceClient
from pymilvus import Function, FunctionType, MilvusClient
from retriever import emb_text, embedding_search, bm25_search, hybrid_search
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
load_dotenv()

HF_TOKEN = os.environ["HF_TOKEN"]
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "UZConstitution_RAG"

milvus_client = MilvusClient(uri="http://localhost:19530")

ranker = Function(
    name="rrf",
    input_field_names=[],
    function_type=FunctionType.RERANK,
    params={
        "reranker": "rrf", 
        "k": 50
    }
)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME,
    cache_folder=r"C:\AMORE\Development\Python Projects\yet-another-RAG-assistant\src\models_cache"
)

llm_client = InferenceClient(
    api_key=HF_TOKEN,
    timeout=120
)


PROMPT_TEMPLATE = """
Используй следующий контекст для ответа на вопрос.

<context>
{context}
</context>

<question>
{question}
</question>
"""


def generate_answer(question):
    hybrid_res = hybrid_res = hybrid_search(milvus_client, COLLECTION_NAME, question, embedding_model, 
                                            emb_field="embedding", sparse_field="sparse", emb_metric="IP", 
                                            ranker=ranker, output_fields=["text"])

    context = "\n".join([line_with_distance[0] for line_with_distance in hybrid_res])
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    
    completion = llm_client.chat.completions.create(
        model="IlyaGusev/saiga_llama3_8b:featherless-ai",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        extra_body={
            "chat_template_kwargs": {
            "enable_thinking": False
            }
        }
    )
    
    answer = completion.choices[0].message.content

    if answer is None:
        return "МОДЕЛЬ ВЕРНУЛА ПУСТОЙ ОТВЕТ"

    return answer