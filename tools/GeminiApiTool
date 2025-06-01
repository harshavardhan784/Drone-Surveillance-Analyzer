import google.generativeai as genai
from utils.utils import wait_for_file_activation
import os
import pandas as pd
from google.generativeai import upload_file
from google.ai.generativelanguage_v1beta.types import Content, Part
from google.protobuf import struct_pb2
import pandas as pd
import mimetypes
import time
# from config import config

class GeminiApiTool:
    def __init__(self, config: dict):
        self.model_name = config.get("model", "gemini-1.5-pro")
        self.api_key = config.get("api_key")
        self.temperature = config.get("temperature", 0.7)
        self.max_output_tokens = config.get("max_output_tokens", 1024)
        self.timeout = config.get("timeout", 300.0)

        if not self.api_key:
            raise ValueError("API key missing for GeminiVideoAnalyzer")
        genai.configure(api_key=self.api_key)

        self.model = genai.GenerativeModel(self.model_name)

    
    def run(self, video_path: str, system_prompt, user_prompt):
        """
        Call Gemini API. If stream=True, yields partial responses.
        """

        if not os.path.exists(video_path):
            print(f"Video file '{video_path}' not found.")
            return "Video file not found.", None
        
        if not self.api_key:
            raise ValueError("API key missing for GeminiApiTool")


        
        full_prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt

        try:
            video_file = upload_file(path=video_path, display_name=os.path.basename(video_path))
            print(f"Uploading: {video_file.display_name}... Status: {video_file.state.name}")

            video_file = wait_for_file_activation(video_file)

            file_part = Part()
            file_part.file_data.mime_type = mimetypes.guess_type(video_path)[0] or "video/mp4"
            file_part.file_data.file_uri = video_file.uri

            parts = [file_part, Part(text=full_prompt)]
            content = Content(role="user", parts=parts)

            response = self.model.generate_content(
                contents=[content],
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_output_tokens,
                },
                request_options={"timeout": self.timeout}
            )


            if response.candidates:
                return response.candidates[0].content.parts[0].text, response
            else:
                return "No content generated.", response

        except Exception as e:
            return f"Error during Gemini API call: {e}", None
