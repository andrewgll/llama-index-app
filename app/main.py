import logging
import os
import pathlib
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import json
import debugpy
import openai
import yaml

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from fastapi.responses import Response as FASTAPIResponse

from llama_index import (
    Document,
    ServiceContext,
    VectorStoreIndex,
    PromptHelper,
    MockEmbedding,
)
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response.schema import Response
from llama_index.storage import StorageContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.llms import OpenAI, ChatMessage
from llama_index.embeddings import OpenAIEmbedding
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index import Prompt, set_global_service_context
from llama_index.llms import MockLLM
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index.memory import ChatMemoryBuffer

import tiktoken
import chromadb
from chromadb.utils import embedding_functions
from config.env import get_settings

with pathlib.Path(__file__).parent.joinpath("../configs/logging.yaml").open() as config:
    logging_config = yaml.load(config, Loader=yaml.FullLoader)

logging.config.dictConfig(logging_config)
env = get_settings()

openai.api_key = env.OPENAI_API_KEY

app = FastAPI()

static_path = Path(__file__).parent / "static"
logging.info(static_path)
templates = Jinja2Templates(directory=static_path)

app.add_middleware(
    CORSMiddleware,
    # HTTPSRedirectMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=env.OPENAI_API_KEY, model_name="text-embedding-ada-002"
)

if env.CHROMADATABASE_HOST:
    db2 = chromadb.HttpClient(host=env.CHROMADATABASE_HOST, port=8000)
else:
    db2 = chromadb.PersistentClient(path="./chroma_db")

llm = OpenAI(temperature=0.2, model="gpt-3.5-turbo")

embed_model = OpenAIEmbedding(embed_batch_size=42)

prompt_helper = PromptHelper(
    context_window=4096, num_output=256, chunk_overlap_ratio=0.1, chunk_size_limit=None
)

service_context = ServiceContext.from_defaults(
    llm=llm, embed_model=embed_model, prompt_helper=prompt_helper
)


chroma_collection = db2.get_or_create_collection(
    "collection", embedding_function=openai_ef
)

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(
    [],
    storage_context=storage_context,
    service_context=service_context,
    similarity_top_k=5,
)


def api_key_validation(request: Request):
    api_key = request.headers.get("Authorization")

    if not api_key or not api_key.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid API key")

    api_key = api_key.replace("Bearer ", "")
    if api_key not in env.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


if os.getenv("ENVIRONMENT") == "development":
    logging.getLogger("app").setLevel("DEBUG")
    debugpy.listen(("0.0.0.0", 5000))


def create_dict_from_arrays(ids, texts):
    if len(ids) != len(texts):
        raise ValueError("Both arrays must have the same length.")

    result_dict = {ids[i]: texts[i] for i in range(len(ids))}
    return result_dict


@app.get("/documents")
def handle_root(api_key: str = Depends(api_key_validation)):
    result = create_dict_from_arrays(
        chroma_collection.get()["ids"], chroma_collection.get()["documents"]
    )
    return result


@app.delete("/documents/{document_id}")
def handle_root(document_id: str, api_key: str = Depends(api_key_validation)):
    try:
        chroma_collection.delete(ids=[document_id])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {e}")
    return FASTAPIResponse(status_code=200)


@app.get("/healthz")
def handle_root():
    return "200"


@app.get("/documents/{document_id}")
def handle_root(document_id: str, api_key: str = Depends(api_key_validation)):
    return index.storage_context.docstore.get_document(document_id)


@app.post("/documents")
async def slack_events(request: Request):
    try:
        body = await request.body()
        decoded_body = body.decode()
        logging.log(level=logging.INFO, msg=decoded_body)
        if decoded_body:
            data = json.loads(decoded_body)
            doc = Document(text=data["query"])
            index.insert(doc)
            return FASTAPIResponse(status_code=200)
    except HTTPException as e:
        raise HTTPException(
            status_code=e.status_code, detail=f"Something went wrong: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {e}")


@app.put("/documents/{document_id}")
async def handle_root(request: Request, document_id: str):
    try:
        body = await request.body()
        decoded_body = body.decode()
        logging.log(level=logging.INFO, msg=decoded_body)
        if decoded_body:
            data = json.loads(decoded_body)
            query = data.get("query")
            logging.log(level=logging.INFO, msg=query)
            chroma_collection.update(ids=document_id, documents=query)
            return FASTAPIResponse(status_code=200)
    except HTTPException as e:
        raise HTTPException(
            status_code=e.status_code, detail=f"Something went wrong: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {e}")


memory = ChatMemoryBuffer.from_defaults(token_limit=1500)


@app.post("/message")
async def message(request: Request, api_key: str = Depends(api_key_validation)):
    try:
        body = await request.body()
        decoded_body = body.decode()
        logging.log(level=logging.INFO, msg=decoded_body)
        if decoded_body:
            data = json.loads(decoded_body)
            query = data.get("query")

            query_engine = index.as_query_engine()

            response = query_engine.query(query)
            logging.log(level=logging.INFO, msg=response)
            response_content = {"text": str(response)}
            return JSONResponse(content=response_content, status_code=200)
        else:
            raise HTTPException(status_code=400, detail="Please specify your message.")
    except HTTPException as e:
        raise HTTPException(
            status_code=e.status_code, detail=f"Something went wrong: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {e}")

