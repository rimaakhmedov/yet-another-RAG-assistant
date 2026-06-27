from huggingface_hub import InferenceClient
from pymilvus import Function, FunctionType, MilvusClient
from retriever import emb_text, embedding_search, bm25_search, hybrid_search
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()
hf_token = os.environ["HF_TOKEN"]

embedding_model_name = "intfloat/multilingual-e5-small"

milvus_client = MilvusClient(uri="http://localhost:19530")
collection_name = "UZConstitution_RAG"
question = "Кто на референдуме выступает от имени народа"

ranker = Function(
    name="rrf",
    input_field_names=[],
    function_type=FunctionType.RERANK,
    params={
        "reranker": "rrf", 
        "k": 50
    }
)


def generate_answer(question, collection_name, embedding_model, milvus_client, ranker):
    hybrid_res = hybrid_search(milvus_client, collection_name, question, 
                           embedding_model, emb_field="embedding", sparse_field="sparse", 
                           emb_metric="IP", ranker=ranker, output_fields=["text"])

    context = "\n".join([line_with_distance[0] for line_with_distance in hybrid_res])
    PROMPT = """
        Используй следующие части информации закрытые в теге <context> для ответа на вопрос в теге <question>.
        <context>
        {context}
        </context>
        <question>
        {question}
        </question>
        """

    llm_client = InferenceClient(
        api_key=os.environ["HF_TOKEN"],
        timeout=120
    )
    prompt = PROMPT.format(context=context, question=question)
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

    print(completion.choices[0].message.content)
    
       
def main():
    print("Загрузка модели")
    embedding_model = SentenceTransformer(
        embedding_model_name,
        cache_folder="src\models_cache"
    )
    
    print("Проверка соединения Milvus")
    if not milvus_client.has_collection(collection_name):
        raise ValueError(f"Collection {collection_name} does not exist")

    print("Генерация ответа")
    generate_answer(
        question=question,
        collection_name=collection_name,
        embedding_model=embedding_model,
        milvus_client=milvus_client,
        ranker=ranker
    )


if __name__ == "__main__":
    main()