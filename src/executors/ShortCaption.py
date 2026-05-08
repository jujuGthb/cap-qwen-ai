"""
Qwen OpenRouter Executor: Short Caption
"""

import os
import sys
import base64
import cv2
from openai import OpenAI

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.QwenOpenRouter.src.utils.response import build_response_caption
from capsules.QwenOpenRouter.src.models.PackageModel import PackageModel

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ShortCaption(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        self.api_key = self.request.get_param("inputApiKey")
        self.model_version = self.request.get_param("inputModelVersion")
        self.temperature = self.request.get_param("inputTemperature")
        self.max_tokens = self.request.get_param("maxTokens")
        self.max_concurrent_requests = self.request.get_param("maxConcurrentRequests")
        self.image_selector = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _build_messages(self, base64_image):
        return [
            {
                "role": "system",
                "content": (
                    "You act as an image caption model. "
                    "Your task is to provide a short, concise description of the image. "
                    "Caption should be short."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            },
        ]

    def run(self):
        img = Image.get_frame(img=self.image_selector, redis_db=self.redis_db)

        success, encoded_image = cv2.imencode('.jpg', img.value)
        if not success:
            raise RuntimeError("Failed to encode image for API")

        base64_image = base64.b64encode(encoded_image).decode('utf-8')
        messages = self._build_messages(base64_image)

        try:
            client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_version,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            if response.choices is None:
                raise RuntimeError("OpenRouter returned no choices in response.")

            self.qwen_text = response.choices[0].message.content or ""
            if not self.qwen_text:
                raise ValueError("Qwen API returned empty content.")

            self.qwen_classes = []

        except Exception as e:
            self.qwen_text = f"API Error: {str(e)}"
            self.qwen_classes = []

        return build_response_caption(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()