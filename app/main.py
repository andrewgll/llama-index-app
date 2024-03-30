import io
import logging
import os
import pathlib

import json
import debugpy
import openai
from pymilvus import MilvusClient
import yaml

from pymilvus import Collection
from fastapi import FastAPI, File, Request, HTTPException, Depends, UploadFile
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pathlib import Path
from fastapi.responses import Response as FASTAPIResponse

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pdfminer.high_level import extract_text
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import Document, ServiceContext, VectorStoreIndex, PromptHelper
from llama_index.core import StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.core.extractors import TitleExtractor
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import get_response_synthesizer
from llama_index.core import Prompt
from llama_index.core.schema import MetadataMode
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)
from llama_index.core.postprocessor import SentenceEmbeddingOptimizer

from app.config.env import get_settings

with pathlib.Path(__file__).parent.joinpath("../configs/logging.yaml").open() as config:
    logging_config = yaml.load(config, Loader=yaml.FullLoader)

if os.getenv("ENVIRONMENT") == "DEBUG":
    logging.getLogger("app").setLevel("DEBUG")
    debugpy.listen(("0.0.0.0", 5001))
    debugpy.wait_for_client()

logging.config.dictConfig(logging_config)
env = get_settings()

if not os.path.exists("/app/model_bin/BAAI/bge-reranker-large"):
    model = AutoModelForSequenceClassification.from_pretrained(
        "BAAI/bge-reranker-large"
    )
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-large")
    tokenizer.save_pretrained("/app/model_bin/BAAI/bge-reranker-large")
    model.save_pretrained("/app/model_bin/BAAI/bge-reranker-large")

rerank = FlagEmbeddingReranker(
    model="/app/model_bin/BAAI/bge-reranker-large", top_n=5, use_fp16=True
)

openai.api_key = env.OPENAI_API_KEY

app = FastAPI()
static_path = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=static_path)

app.add_middleware(
    CORSMiddleware,
    # HTTPSRedirectMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

text_qa_template = (
    "You are a helpful assistant to the company. You will receive contextual information, which you can find below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "With this information"
    "answer the question: {query_str}\n"
    "Try to use the context information as much as possible to answer the question. \n"
)
text_qa_template = Prompt(text_qa_template)

refine_template_str = (
    "Act as a customer service answer the question succinctly: {query_str}\n"
    "Combine with existing answer: {existing_answer}\n"
    "------------\n"
    "And refine with further context: {query_str}\n"
    "------------\n"
    "{context_msg}\n"
    "------------\n"
    "Be brief, polite, and respectful when responding to customer inquiries"
)
refine_template = Prompt(refine_template_str)

# IP address of docker bridge
vector_store = MilvusVectorStore(uri="http://172.17.0.1:19530", dim=1536)
# If specified db host url
if env.MILVUSVECTORSTORE:
    vector_store = MilvusVectorStore(uri=env.MILVUSVECTORSTORE, dim=1536)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

milvus_client: MilvusClient = vector_store.client

llm = OpenAI(temperature=0.2, model="gpt-3.5-turbo")

prompt_helper = PromptHelper(
    context_window=4096, num_output=256, chunk_overlap_ratio=0.1, chunk_size_limit=None
)

service_context = ServiceContext.from_defaults(
    llm=llm,
    prompt_helper=prompt_helper,
    embed_model=OpenAIEmbedding(embed_batch_size=42),
)

similarity_postprocessor = SimilarityPostprocessor(similarity_cutoff=0.75)
sentece_postprocessor = SentenceEmbeddingOptimizer(
    embed_model=service_context.embed_model,
    percentile_cutoff=0.75,
    threshold_cutoff=0.7,
)

TITLE_NODE_TEMPLATE = """\
Context: {context_str}. Give a very short and comprehensive title that summarizes this text using 2 words. Title: """
DEFAULT_TITLE_COMBINE_TEMPLATE = """\
{context_str}. Based on the above candidate titles and content, \
choose very short, maximum 2 words title for this document. Title: """
pipeline = IngestionPipeline(
    transformations=[
        SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=service_context.embed_model,
        ),
        TitleExtractor(
            llm=llm,
            nodes=7,
            metadata_mode=MetadataMode.EMBED,
            num_workers=8,
            node_template=TITLE_NODE_TEMPLATE,
        ),
        OpenAIEmbedding(embed_batch_size=42),
    ],
    vector_store=vector_store,
)


def api_key_validation(request: Request):
    api_key = request.headers.get("Authorization")

    if not api_key or not api_key.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Invalid API key")

    api_key = api_key.replace("Bearer ", "")
    if api_key not in env.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


@app.get("/documents", response_class=JSONResponse)
def handle_get_documents(api_key: str = Depends(api_key_validation)):
    collection = Collection("llamacollection")  # Get an existing collection.
    data = {
        "schema": collection.schema,
        "description": collection.description,
        "name": collection.name,
        "is_empty": collection.is_empty,
        "num_entities": collection.num_entities,
        "primary_field": collection.primary_field,
        "partitions": [partition.dict() for partition in collection.partitions],
        "indexes": collection.indexes,
    }

    return data


@app.get("/healthz")
def handle_health():
    return "200"


@app.post("/documents")
async def handle_post_documents(request: Request, pdf_file: UploadFile = File(...)):
    try:
        pdf_content = await pdf_file.read()

        pdf_file_object = io.BytesIO(pdf_content)

        text = extract_text(pdf_file_object)
        logging.log(level=logging.INFO, msg=text)
        if text:
            doc = Document(text=text)
            pipeline.run(documents=[doc], show_progress=True, num_workers=8)
            return FASTAPIResponse(status_code=200)
    except HTTPException as e:
        raise HTTPException(
            status_code=e.status_code, detail=f"Something went wrong: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {e}")


@app.post("/message")
async def handle_post_message(
    request: Request, api_key: str = Depends(api_key_validation)
):
    try:
        body = await request.body()
        decoded_body = body.decode()
        logging.log(level=logging.INFO, msg=decoded_body)
        if decoded_body:
            data = json.loads(decoded_body)
            query = data.get("query")

            response_synthesizer = get_response_synthesizer(
                service_context=service_context,
                # text_qa_template=text_qa_template,
                # refine_template=refine_template,
                response_mode="compact",
            )
            embedded_index = VectorStoreIndex(
                [], service_context=service_context, storage_context=storage_context
            )

            retriever = embedded_index.as_retriever(
                similarity_top_k=10,
            )
            splitter = SentenceSplitter(
                chunk_size=1024,
                chunk_overlap=20,
                include_metadata=False
            )
            nodes = await retriever.aretrieve(query)

            filtered_nodes_stage_1 = similarity_postprocessor.postprocess_nodes(nodes)

            splitted_nodes = await splitter.acall(filtered_nodes_stage_1)

            filtered_nodes_stage_2 = sentece_postprocessor.postprocess_nodes(
                splitted_nodes
            )

            filtered_nodes_stage_3 = rerank.postprocess_nodes(filtered_nodes_stage_2)

            output_index = VectorStoreIndex(
                filtered_nodes_stage_3,
                service_context=service_context,
                similarity_top_k=3,  # default value if not specified
                response_synthesizer=response_synthesizer,
            )

            query_engine = output_index.as_query_engine(
                chat_mode="best",
                verbose=True,
                response_synthesizer=response_synthesizer,
            )

            text = await query_engine.aquery(query)

            response = text.response
            logging.log(level=logging.INFO, msg=text)
            response_content = {"text": response}
            return JSONResponse(content=response_content, status_code=200)
        else:
            raise HTTPException(status_code=400, detail="Please specify your message.")
    except HTTPException as e:
        raise HTTPException(
            status_code=e.status_code, detail=f"Something went wrong: {e.detail}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {e}")
