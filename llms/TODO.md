# TODO


- Batch inference
    - Google, OpenAI: implement a la Anthropic example using Async; allow param for multiple API keys; one ClientManager and executor pool per process
    - For HuggingFace

- Video Inputs
    - Add prompting for video inputs
    - Create `any_to_video` similarly to `any_to_pil` to homogeneize format


- Client Managers `num_concurrent_processes`: 
    - Logic to set number of concurrent processes using an API key is incomplete and only works for threads in the same process
    - Needs to write to shared file
    - Add logic to write to the `api_keys.json` file the number of processes using the api_key
    - Similar for `num_retries` if wants max retries among all processes using the API keys (current each process will try `max_times` for each key)



- Revise and finish `tokenizers.py` implementation