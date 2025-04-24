## Summary
This is a set of tools to make it easy to send inference calls to MLLMs across different:
- Model providers (e.g.: OpenAI , HuggingFace) and model families, 
- Deployment engines (e.g.: VLLM, OpenRouter, local uvicorn hosted server), 
- API modes (OpenAI chat completions and response API)

Using a uniform format for messages and generation args and automatic handling of errors, API keys, etc.

The idea is to receive a list of multimodal generations from any model by simply providing a list of raw inputs (in flexible format) and a dict of generation arguments, abstracting away any other headhaches.

Like below:
```
inputs = [
    PIL_image, 
    text, 
    {"role": "system", "text": "You are a helpful assistant"}, 
    ["image 1", "test_llm/dog.png"],
    numpy_image,
    url_image,
    video,
]

gen_args = {
    "model": "gemini-2.0-flash-001",
    "temperature": 1.0,
    "top_p": 0.9,
    "top_k": 40,
    "num_generations": 2
}

api_responses, model_messages = call_llm(gen_kwargs, inputs)
```

Main components:
1) Simple and flexible input-output format. In essence:
- (i) Provide a list of raw inputs (PIL/Numpy/URL/path images, text, videos, dictionaries) 
- (ii) Provide a dictionary of generation arguments (temperature, num_generations, etc).
- (iii) Receive a list of outputs (text, images, function calls, etc) back.

2) Automatic logging relevant information, handling errors, etc
Behind the scenes, during an inference call the tools also:
- Perform retries with exponential backoff handling provider-specific errors
- Manages and balance load among API keys in case of more than one
- Dynamically upload prompts to the cloud in case of large payloads and automatically retry
- Logging visual representations of the conversation
- Logging token usage
- Log timings for specific calls
- etc

3) Homogeneous representation for elements of a conversation and generation arguments.
- Any transformations needed to inputs to necessary for a provider, model, engine is done behind the scenes
- Similarly, any unsupported parameters for a model or provider are pruned away

## Setup
- Place the `llms` and `utils` in the root folder of your project
- Install libraries in requirements.txt
- Optionally install Flash Attention for fastr Hugging Face inference: `pip install flash-att`
- Place your api keys in the `api_keys.json` files. 
    - You can hide them or chage where they are located; make sure to also change the `llms/constants/constants.py` paths. 
- Check the `examples/examples.ipynb`

## Note
NOTE: this is a package under construction. Most of the relevant utilities are working, however:
- You may need to tweak the relative paths. Recommended: place both `utils` and `llms` in the root directory of the project.
- HuggingFace models have specific quirks, and it is impossible to foresee all them. The tools will do a best effort, but may need to change the code in `llms/providers/hugging_face` to support a new model (typically one or two lines).
- Repo may have some bugs and non-implemented arguments for particular providers/models
- More documentation will be added in the future
- Video, function calling, computer use are not fully tested.

