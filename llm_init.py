#from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.embeddings import FastEmbedEmbeddings
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# --- 1. Define your Embedding Model (BGE-Large with FastEmbed) ---
embeddings = FastEmbedEmbeddings(model_name=os.getenv("embedding_model"))





# llm = ChatOpenAI(
# openai_api_base="http://192.168.18.100:8000/v1",
# openai_api_key="EMPTY",
# model_name="gpt-oss-20b"
# )

# eval_llm = ChatOpenAI(
#    openai_api_base="http://192.168.18.100:8000/v1",
#    openai_api_key="EMPTY",
#    model_name="gpt-oss-20b"
# )

# llm1 = ChatOpenAI(
#   openai_api_base="http://192.168.18.100:8000/v1",
#   openai_api_key="EMPTY",
#   model_name="gpt-oss-20b"
# )

from langchain_openai import ChatOpenAI

# Main LLM
#llm = ChatOpenAI(
    #openai_api_base="https://api.groq.com/openai/v1",
    # openai_api_key="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",   # 🔑 Put your Groq API key here
 #   model_name="openai/gpt-oss-20b"
#)

# Evaluation LLM
#eval_llm = ChatOpenAI(
   # openai_api_base="https://api.groq.com/openai/v1",
  #  openai_api_key="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",
 #   model_name="openai/gpt-oss-20b"
#)

# Another instance if needed
#llm1 = ChatOpenAI(
 #   openai_api_base="https://api.groq.com/openai/v1",
  #  openai_api_key="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",
   # model_name="openai/gpt-oss-20b"
#)













