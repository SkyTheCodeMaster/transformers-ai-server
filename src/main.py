from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from aiohttp import web
from transformers import GPTNeoXForCausalLM, AutoTokenizer

if TYPE_CHECKING:
  from typing import Coroutine, Any
  from aiohttp.web import Request, Response

fmt = "[%(filename)s][%(asctime)s][%(levelname)s] %(message)s"
datefmt = "%Y/%m/%d-%H:%M:%S"

logging.basicConfig(
  handlers = [
    logging.StreamHandler()
  ],
  format= fmt,
  datefmt = datefmt,
  level=logging.INFO,
)

LOG = logging.getLogger()

DEVICE = "cuda:0"
MODEL = "EleutherAI/pythia-2.8b-deduped"

routes = web.RouteTableDef()

# this is more of a chatbot type
LOG.info("Initializing model...")
model = GPTNeoXForCausalLM.from_pretrained(MODEL,revision="step3000").to(DEVICE)
LOG.info("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL,revision="step3000")

def _generate(text: str, max_length: int) -> str:
  LOG.info(f"Tokenizing text: '{text[:70]}'...")
  inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
  LOG.info("Generating...")
  tokens = model.generate(**inputs, max_length=max_length)
  LOG.info("Decoding...")
  return tokenizer.decode(tokens[0])

async def generate_text(text: str, max_length) -> Coroutine[Any, Any, str]:
  loop = asyncio.get_running_loop()
  response = await loop.run_in_executor(None, lambda: _generate(text, max_length))
  return response

@routes.post("/generate")
async def post_generate(request: Request) -> Response:
  body = await request.json()
  data = body.get("text",None)
  max_length = body.get("max_length",150)
  if not data:
    return web.Response(text="no text passed",status=400)
  
  LOG.info(f"Processing request...")
  
  output = await generate_text(data,max_length)

  response = web.Response(body=output,status=200,content_type="text/plain")
  return response

app = web.Application()
app.add_routes(routes)

LOG.info("Starting webserver...")
web.run_app(app,host="0.0.0.0",port=12503) # type: ignore