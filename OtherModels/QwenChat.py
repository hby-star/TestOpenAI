import json
import time

from openai import OpenAI

import Utils
from SecretKey import QwenKey


client = OpenAI(
    api_key=QwenKey,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

chat_time = 0
completion_choice = "text"

if completion_choice == "text":
    start_time = time.time()
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ],
        stream=False
    )
    end_time = time.time()
    chat_time = end_time - start_time
    print(f"Time taken: {end_time - start_time}")


"""
Response
"""


if completion_choice:
    print(completion)
    # Normalize the response
    normalized_response = {
        "time": chat_time,
        "id": completion.id,
        "choices": [
            {
                "finish_reason": choice.finish_reason,
                "index": choice.index,
                "logprobs": choice.logprobs,
                "message": {
                    "content": choice.message.content,
                    "refusal": choice.message.refusal,
                    "role": choice.message.role,
                    "audio": Utils.serialize_audio(choice.message.audio) if choice.message.audio else None,
                    "function_call": choice.message.function_call,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                            "type": tool_call.type,
                        } for tool_call in choice.message.tool_calls
                    ] if choice.message.tool_calls else None
                }
            } for choice in completion.choices
        ],
        "created": completion.created,
        "model": completion.model,
        "object": completion.object,
        "service_tier": completion.service_tier,
        "system_fingerprint": completion.system_fingerprint,
        "usage": {
            "completion_tokens": completion.usage.completion_tokens,
            "prompt_tokens": completion.usage.prompt_tokens,
            "total_tokens": completion.usage.total_tokens,
        }
    }

    # Save the normalized response to response.json
    with open('../Response/qwen_response.json', 'w') as f:
        json.dump(normalized_response, f, indent=4)
