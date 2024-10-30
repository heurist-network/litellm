import os, types
import json
import random
import requests
import time
from typing import Callable, Optional, Literal, List
from litellm.types.utils import Function,ChatCompletionMessageToolCall, ModelResponse,Usage,Choices,Message
import litellm
import httpx
from litellm.asyncsseclient import asyncsseclient
from .prompt_templates.factory import prompt_factory, custom_prompt
from openai.types.chat import ChatCompletionMessage, ChatCompletionMessageToolCall
from pydantic import BaseModel

class HeuristConfig:
    def __init__(self):
        self.model_configs = self.fetch_model_configs()

    def fetch_model_configs(self):
        model_config_url = "https://raw.githubusercontent.com/heurist-network/heurist-models/main/models.json"
        response = requests.get(model_config_url)
        return json.loads(response.text)

    def get_llm_models(self):
        return [
            {
                "id": model["name"],
                "object": "model",
            }
            for model in self.model_configs
            if model.get("type", "").startswith("llm")
        ]

    def get_model_config(self, model):
        model = normalize_model_id(model)
        return next((config for config in self.model_configs if config["name"] == model), None)

    @classmethod
    def get_config(cls):
        return cls()



APP_ID = "heurist-llm-gateway"
end_of_stream = "[DONE]"

# TODO: support these
default_priority = 1
default_deadline = 60

def normalize_model_id(model_id):
    if model_id == "mistralai/mixtral-8x7b-instruct-v0.1":
        return "mistralai/mixtral-8x7b-instruct"
    if model_id == "mistralai/mistral-7b-instruct-v0.2":
        return "mistralai/mistral-7b-instruct"
    return model_id

def get_random_job_id():
    # get 10 random letters and numbers
    return APP_ID + "".join([chr(random.randint(97, 122)) for _ in range(10)])

def submit_job(api_base, job_id, model_input, model_id, api_key, temperature, max_tokens, tools, extra_body, use_stream=True):
    url = api_base + "/submit_job"
    print("submitting job", api_base, job_id, model_input, model_id, api_key, temperature, max_tokens, tools, extra_body, use_stream)
    job = {
        "job_id": job_id,
        "model_input": {
            "LLM": {
                "prompt": model_input,
                "use_stream": use_stream,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        },
        "model_type": "LLM",
        "model_id": normalize_model_id(model_id),
        "deadline": default_deadline,
        "priority": default_priority,
    }
    
    if tools:
        job["model_input"]["LLM"]["tools"] = json.dumps(tools)
    if extra_body:
        job["model_input"]["LLM"]["extra_body"] = json.dumps(extra_body)
    
    headers = {
        "Authorization" : f"Bearer {api_key}"
    }
    response = requests.post(url, json=job, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Error submitting job {job_id}: {response.text}")

    if use_stream:
        return api_base + f"/stream/{job_id}"
    else:
        return response.text
    
async def handle_stream(stream_url):
    client = asyncsseclient(stream_url)
    async for event in client:
        if end_of_stream in event.data:
            break
        if event.data:
            yield event.data

def completion(
    model: str,
    messages: list,
    api_base: str,
    model_response: ModelResponse,
    print_verbose: Callable,
    logging_obj,
    api_key,
    encoding,
    custom_prompt_dict={},
    optional_params=None,
    litellm_params=None,
    logger_fn=None,
    heurist_model_config=None
):
    prompt = prompt_factory(model=model, messages=messages)

    user_api_key = litellm_params['metadata'].get('user_api_key', None)

    ## COMPLETION CALL
    model_response.created = int(time.time())


    temperature = optional_params.get("temperature", 0.75)
    
    if ("mistral" in model or "mixtral" in model) and temperature < 0.01:
        temperature = 0.01
    
    max_tokens = optional_params.get("max_tokens", 500)

    tools = optional_params.get("tools", None)

    # Fetch model configuration
    model_config = heurist_model_config

    if model_config is None:
        print(f"Warning: Model {model} not found in configuration")
        model_config = {}

    # Check if tools are supported
    if tools is not None and not model_config.get("tool_call_parser", False):
        print(f"Warning: Model {model} does not support tools: {tools}")
        tools = None

    # check if have redirect field
    redirect = model_config.get("redirect", None)
    if redirect:
        model = model_config["redirect"]
    
    # Extract guided parameters and other extra body fields
    extra_body = {}
    guided_params = ["guided_json", "guided_regex", "guided_choice", "guided_grammar"]
    other_params = ["echo", "add_generation_prompt", "include_stop_str_in_output", "guided_decoding_backend", "guided_whitespace_pattern"]
    
    for param in guided_params + other_params:
        if param in optional_params:
            extra_body[param] = optional_params[param]
    
    # If extra_body is empty, set it to None
    extra_body = extra_body if extra_body else None
    
    
    if "stream" in optional_params and optional_params["stream"] == True:
        return handle_stream(submit_job(api_base, get_random_job_id(), prompt, model, user_api_key, temperature, max_tokens, tools, extra_body, use_stream=True))
    else:
        result = submit_job(api_base, get_random_job_id(), prompt, model, user_api_key, temperature, max_tokens, tools, extra_body, use_stream=False)
        model_response.ended = int(time.time())

        print_verbose(f"raw model_response: {result}")

        try:
            choices = json.loads(result)
            if isinstance(choices, list):
                # Replace the existing choices with the new ones
                model_response.choices = []
                for idx, choice in enumerate(choices):
                    message_data = choice.get('message', {})
                    message_content = message_data.get('content')
                    tool_calls_data = message_data.get('tool_calls')
                    
                    tool_calls = None
                    if tool_calls_data:
                        tool_calls = [
                            {
                                "id": tool_call.get('id'),
                                "type": tool_call.get('type'),
                                "function": {
                                    "name": tool_call.get('function', {}).get('name'),
                                    "arguments": tool_call.get('function', {}).get('arguments')
                                }
                            }
                            for tool_call in tool_calls_data
                        ]

                    message = Message(
                        content=message_content,
                        role=message_data.get('role', 'assistant'),
                        tool_calls=tool_calls
                    )
                    
                    choice_obj = Choices(
                        index=idx,
                        message=message,
                        finish_reason=choice.get('finish_reason')
                    )
                    model_response.choices.append(choice_obj)

                model_response.model = "heurist/" + model
                
                # If usage information is available in the response, update it
                if choices and isinstance(choices[0], dict) and 'usage' in choices[0]:
                    usage_data = choices[0]['usage']
                    model_response.usage = Usage(
                        prompt_tokens=usage_data.get('prompt_tokens', 0),
                        completion_tokens=usage_data.get('completion_tokens', 0),
                        total_tokens=usage_data.get('total_tokens', 0)
                    )
                else:
                    # If no usage information, keep the default values
                    model_response.usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
                
                return model_response

        except json.JSONDecodeError:
            print_verbose(f"Failed to parse JSON response: {result}")
            # Fallback to the original behavior if JSON parsing fails
            if len(result) > 1:
                model_response.choices[0].message.content = result
            model_response.usage = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    model_response.model = "heurist/" + model
    return model_response


