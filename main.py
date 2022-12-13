from typing import Union

from pydantic import BaseModel
from fastapi import FastAPI
from codegen import Codegen

app = FastAPI(
    title="FastAPI Codegen",
    description="Local code suggestions using CodeGen",
    version="1.0",
    openapi_url="/api/openapi.json",
)

codegen = Codegen()

class CodeRequest(BaseModel):
    input_text: str

class CodeResponse(BaseModel):
    response_text: str

@app.post("/", response_model=CodeResponse)
async def code_request(
    code_request: CodeRequest
    ):
    """
    Codegen Request
    """
    suggestion = await codegen.get_suggestion(code_request.input_text)
    return CodeResponse(response_text=suggestion)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)