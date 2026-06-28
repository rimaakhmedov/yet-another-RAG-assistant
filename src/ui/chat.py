import gradio as gr
import time
from dotenv import load_dotenv
import os
import sys
from pathlib import Path
from pymilvus import Function, FunctionType, MilvusClient
from sentence_transformers import SentenceTransformer
sys.path.append(str(Path(__file__).resolve().parent.parent))
from generation import generate_answer


def respond(message, chat_history):
        bot_message = generate_answer(message)
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        time.sleep(2)
        return "", chat_history
    
    
with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch()