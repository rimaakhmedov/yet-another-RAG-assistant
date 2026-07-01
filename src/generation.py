from pymilvus import Function, FunctionType, MilvusClient
from retriever import emb_text, embedding_search, bm25_search, hybrid_search
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os
load_dotenv()


EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
COLLECTION_NAME = "UZConstitution_RAG"
MODEL_NAME = "qwen/qwen3-4b"

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
    cache_folder=r""
)

client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
print("OpenAI API клиент инициализирован")

PROMPT_TEMPLATE = """
Используй контекст для ответа на вопрос.

При ответе обязательно указывай:
- ТОЧНОЕ название документа/файла;
- номер страницы, если информация найдена в контексте.

<context>
{context}
</context>

<question>
{question}
</question>
"""

def generate_answer(question):
    hybrid_res = hybrid_search(milvus_client, COLLECTION_NAME, question, embedding_model, 
                                            emb_field="embedding", sparse_field="sparse", emb_metric="IP", 
                                            ranker=ranker, output_fields=["text", "document_name", "page_number"])

    context_parts = []

    for chunk in hybrid_res:
        context_parts.append(
            f"""
        Источник: {chunk['document_name']}
        Страница: {chunk['page_number']}

        {chunk['text']}
        """
            )

    context = "\n\n".join(context_parts)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=2048,
            temperature=0.5
        )
        
        answer = response.choices[0].message.content
        
        if not answer or answer.strip() == "":
            return "МОДЕЛЬ ВЕРНУЛА ПУСТОЙ ОТВЕТ"
        
        return answer
    
    except Exception as e:
        return f"Ошибка при обращении к OpenAI API: {str(e)}"


# print(generate_answer('Кто на референдуме выступает от имени народа'))