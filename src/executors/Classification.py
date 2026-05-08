"""
Qwen OpenRouter Executor: Single-Label Classification
"""

import os
import sys
import base64
import json
import cv2
from openai import OpenAI

sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../'))

from sdks.novavision.src.media.image import Image
from sdks.novavision.src.base.capsule import Capsule
from sdks.novavision.src.helper.executor import Executor
from capsules.QwenOpenRouter.src.utils.response import build_response_classification
from capsules.QwenOpenRouter.src.models.PackageModel import PackageModel

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class Classification(Capsule):
    def __init__(self, request, bootstrap):
        super().__init__(request, bootstrap)
        self.request.model = PackageModel(**(self.request.data))

        self.classes = self.request.get_param("inputClasses")
        self.api_key = self.request.get_param("inputApiKey")
        self.model_version = self.request.get_param("inputModelVersion")
        self.temperature = self.request.get_param("inputTemperature")
        self.max_tokens = self.request.get_param("maxTokens") or 500
        self.max_concurrent_requests = self.request.get_param("maxConcurrentRequests")
        self.image_selector = self.request.get_param("inputImage")

    @staticmethod
    def bootstrap(config: dict) -> dict:
        return {}

    def _build_messages(self, base64_image):
        serialised_classes = ", ".join(self.classes) if isinstance(self.classes, list) else (self.classes or "")
        return [
            {
                "role": "system",
                "content": (
                    "You act as single-class classification model. You must provide reasonable predictions. "
                    "You are only allowed to produce JSON document in Markdown ```json``` markers. "
                    'Expected structure of json: {"class_name": "class-name", "confidence": 0.4}. '
                    "`class-name` must be one of the class names defined by user. You are only allowed to return "
                    "single JSON document, even if there are potentially multiple classes. You are not allowed to return list. "
                    "You cannot discuss the result, you are only allowed to return JSON document."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"List of all classes to be recognised by model: {serialised_classes}"},
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

            try:
                clean_text = self.qwen_text.strip()
                if clean_text.startswith("```"):
                    clean_text = clean_text[clean_text.find("\n") + 1:]
                    clean_text = clean_text[:clean_text.rfind("```")].strip()
                parsed = json.loads(clean_text)
                class_name = parsed.get("class_name", "")
                self.qwen_classes = [class_name] if class_name else []
            except (json.JSONDecodeError, KeyError):
                self.qwen_classes = []

        except Exception as e:
            self.qwen_text = f"API Error: {str(e)}"
            self.qwen_classes = []

        return build_response_classification(context=self)


if "__main__" == __name__:
    Executor(sys.argv[1]).run()