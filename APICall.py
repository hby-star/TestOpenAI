import json
import base64
from openai import OpenAI
from SecretKey import SecretKey

client = OpenAI(
    api_key=SecretKey
)

# completion_choice = "text"
completion_choice = "image&text"

"""
Text
"""
if completion_choice == "text":
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        store=True,
        messages=[
            {
                "role": "user",
                "content": "write a haiku about ai"
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "email_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "email": {
                            "description": "The email address that appears in the input",
                            "type": "string"
                        },
                        "additionalProperties": False
                    }
                }
            }
        }
    )


"""
Image & Text
"""
if completion_choice == "image&text":
    image_path = "./images/avatar.jpg"
    def encode_image(_image_path):
        with open(_image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    # Getting the base64 string
    base64_image = encode_image(image_path)

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


# Output
if completion_choice:
    # Normalize the response
    normalized_response = {
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
                    "audio": choice.message.audio,
                    "function_call": choice.message.function_call,
                    "tool_calls": choice.message.tool_calls
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

    # Save the normalized response to response.json
    with open('response.json', 'w') as f:
        json.dump(normalized_response, f, indent=4)
