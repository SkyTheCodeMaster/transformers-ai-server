from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from aiohttp import web
from transformers import GPTNeoXForCausalLM, AutoTokenizer

if TYPE_CHECKING:
  from typing import Coroutine, Any
  from aiohttp.web import Request, Response

DEVICE = "cuda:0"
MODEL = "EleutherAI/pythia-2.8b-deduped"

routes = web.RouteTableDef()

# this is more of a chatbot type
print("Initializing model...")
model = GPTNeoXForCausalLM.from_pretrained(MODEL,revision="step3000").to(DEVICE)
print("Initializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL,revision="step3000")

def _generate(text: str) -> str:
  print(f"Tokenizing text: {text[:70]}...")
  inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
  print("Generating...")
  tokens = model.generate(**inputs)
  print("Decoding...")
  return tokenizer.decode(tokens[0])

async def generate_text(text: str) -> Coroutine[Any, Any, str]:
  loop = asyncio.get_running_loop()
  response = await loop.run_in_executor(None, lambda: _generate(text))
  return response

@routes.post("/generate")
async def post_generate(request: Request) -> Response:
  body = await request.json()
  data = body.get("text",None)
  if not data:
    return web.Response(text="no text passed",status=400)
  
  print(f"Processing request...")
  
  output = await generate_text(data)

  response = web.Response(body=output,status=200,content_type="text/plain")
  return response

app = web.Application()
app.add_routes(routes)

print("Starting webserver...")
web.run_app(app,host="0.0.0.0",port=12503) # type: ignore