import os
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

try:
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
except ImportError:
    print("Warning: vllm not installed or not supported on this OS (Windows).")
    AsyncLLMEngine = None

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '../../configs/pipeline_config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

app = FastAPI(title="Aligned LLM vLLM Server")
config = load_config()

# Global engine variable
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    if AsyncLLMEngine is None:
        print("vLLM is not available. Running in mock mode.")
        return
        
    # We serve the final DPO adapter. vLLM can load PEFT adapters on top of base models.
    # However, it's usually better to merge the PEFT weights into the base model first.
    # Assuming `dpo_adapter_path` has the model or we merged it in a separate script.
    # For demo purposes, we point to base model if adapter is tricky in vllm without merging.
    
    model_path = config['model']['base_model_name_or_path'] # Ideally merged_model_path
    
    print(f"Initializing vLLM Engine with model: {model_path}")
    engine_args = AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=config['serving']['tensor_parallel_size'],
        trust_remote_code=True,
        max_model_len=config['training']['max_seq_length']
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    max_tokens = data.get("max_tokens", 256)
    temperature = data.get("temperature", 0.7)
    
    if engine is None:
        return JSONResponse({"text": "Mock response: vLLM not available. Prompt was: " + prompt})
        
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    request_id = str(os.urandom(8).hex())
    results_generator = engine.generate(prompt, sampling_params, request_id)
    
    final_output = None
    async for request_output in results_generator:
        final_output = request_output
        
    text = final_output.outputs[0].text
    return JSONResponse({"text": text})

if __name__ == "__main__":
    host = config['serving']['host']
    port = config['serving']['port']
    print(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
