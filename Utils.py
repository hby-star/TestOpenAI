import base64
import io

import requests
from pydub import AudioSegment

def encode_image(_image_path):
    with open(_image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def convert_audio_to_format(audio_data, output_format):
    # Decode the base64 audio data
    audio_bytes = base64.b64decode(audio_data)

    # Create an AudioSegment instance from the raw audio data
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes), format="wav")

    # Export the audio to the desired format
    output_file = f"./Response/response.{output_format}"
    audio_segment.export(output_file, format=output_format)

    return output_file

def serialize_audio(audio):
    if audio is None:
        return None

    audio_data_path = convert_audio_to_format(audio.data, "wav")

    return {
        "id": audio.id,
        "expires_at": audio.expires_at,
        "data": audio_data_path,
        "transcript": audio.transcript,
    }

