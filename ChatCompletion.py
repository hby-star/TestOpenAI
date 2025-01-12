import json
import base64
import time

import requests
import Utils
from pydantic import BaseModel
from openai import OpenAI
from SecretKey import OpenAIKey

client = OpenAI(
    api_key=OpenAIKey
)

chat_time = 0
completion_choice = "text"
# completion_choice = "image&text"
# completion_choice = "audio"
# completion_choice = "tools"

"""
Text
"""
if completion_choice == "text":
    start_time = time.time()
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
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
Image & Text
"""
if completion_choice == "image&text":
    image_path = "./images/avatar.jpg"

    # Getting the base64 string
    base64_image = Utils.encode_image(image_path)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )


"""
Audio
"""
if completion_choice == "audio":
    url = "https://openaiassets.blob.core.windows.net/$web/API/docs/audio/alloy.wav"
    response = requests.get(url)
    response.raise_for_status()
    wav_data = response.content
    encoded_string = base64.b64encode(wav_data).decode('utf-8')

    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this recording?"
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_string,
                            "format": "wav"
                        }
                    }
                ]
            },
        ]
    )

"""
Using tools
"""
if completion_choice == "tools":
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                },
            },
        }
    ]

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the weather like in Shanghai today?"}],
        tools=tools,
    )

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
            "completion_tokens_details": {
                "accepted_prediction_tokens": completion.usage.completion_tokens_details.accepted_prediction_tokens,
                "audio_tokens": completion.usage.completion_tokens_details.audio_tokens,
                "reasoning_tokens": completion.usage.completion_tokens_details.reasoning_tokens,
                "rejected_prediction_tokens": completion.usage.completion_tokens_details.rejected_prediction_tokens
            },
            "prompt_tokens_details": {
                "audio_tokens": completion.usage.prompt_tokens_details.audio_tokens,
                "cached_tokens": completion.usage.prompt_tokens_details.cached_tokens
            }
        }
    }

    # Save the normalized response
    with open('Response/openai_response.json', 'w') as f:
        json.dump(normalized_response, f, indent=4)
