from pydantic import validator
from typing import List, Optional, Union, Literal
from sdks.novavision.src.base.model import (
    Package, Image, Inputs, Outputs, Configs, Response, Request, Output, Input, Config
)


class InputImage(Input):
    name: Literal["inputImage"] = "inputImage"
    value: Union[List[Image], Image]
    type: str = "object"

    @validator("type", pre=True, always=True)
    def set_type_based_on_value(cls, value, values):
        value = values.get("value")
        if isinstance(value, list):
            return "list"
        return "object"

    class Config:
        title = "Image"


class OutputText(Output):
    name: Literal["output"] = "output"
    value: Optional[str]
    type: Literal["string"] = "string"

    class Config:
        title = "Output"


class Classes(Output):
    name: Literal["classes"] = "classes"
    value: Union[List[str], str]
    type: str = "object"

    @validator("type", always=True)
    def set_type_based_on_value(cls, val, values):
        val = values.get("value")
        if isinstance(val, list):
            return "list"
        return "object"

    class Config:
        title = "Classes"

class InputOpenRouterApiKey(Config):
    """
    Your OpenRouter API key used to authenticate requests to the Qwen model.
    Obtain this from openrouter.ai under API Keys.
    Keep it private and never expose it in client-side code.
    """
    name: Literal["inputApiKey"] = "inputApiKey"
    value: str = ""
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "OpenRouter API Key"
        json_schema_extra = {"shortDescription": "OpenRouter Key"}


class InputPrompt(Config):
    """
    The custom prompt sent with the image to guide the model's response.
    Use this to specify the task, output format, or focus area.
    """
    name: Literal["inputPrompt"] = "inputPrompt"
    value: str = ""
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Prompt"
        json_schema_extra = {"shortDescription": "User Prompt"}


class InputClasses(Config):
    """
    Enter the list of classes as a JSON array.
    Example: ["cat", "dog", "bird"]
    Used for Classification, Multi-Label, and Object Detection tasks.
    """
    name: Literal["inputClasses"] = "inputClasses"
    value: List[str]
    type: Literal["list"] = "list"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Classes"
        json_schema_extra = {"shortDescription": "Class List"}


class InputOutputStructure(Config):
    """
    A JSON object specifying the expected output structure.
    Each key is a field name and each value is a description of that field.
    Example: {"color": "dominant color in the image", "mood": "overall mood"}
    """
    name: Literal["inputOutputStructure"] = "inputOutputStructure"
    value: str = ""
    type: Literal["string"] = "string"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Output Structure"
        json_schema_extra = {"shortDescription": "JSON Output Schema"}


class TemperatureConfig(Config):
    """
    Controls the randomness of the model output (0.0–2.0).
    Lower values produce more deterministic results.
    Higher values produce more varied and creative responses.
    """
    name: Literal["inputTemperature"] = "inputTemperature"
    value: float = 0.1
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Temperature"
        json_schema_extra = {"shortDescription": "Output Randomness"}


class MaxTokens(Config):
    """
    Maximum number of tokens the model can generate in its response.
    Increase for longer outputs such as detailed captions or structured answers.
    Default is 500.
    """
    name: Literal["maxTokens"] = "maxTokens"
    value: int = 500
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Max Tokens"
        json_schema_extra = {"shortDescription": "Max Output Length"}


class MaxConcurrentRequests(Config):
    """
    Maximum number of API requests to run in parallel when processing a batch of images.
    If not set, falls back to the global Workflows Execution Engine default.
    """
    name: Literal["maxConcurrentRequests"] = "maxConcurrentRequests"
    value: int = 4
    type: Literal["number"] = "number"
    field: Literal["textInput"] = "textInput"

    class Config:
        title = "Max Concurrent Requests"
        json_schema_extra = {"shortDescription": "Parallel Requests"}



class Version35BA3B(Config):
    """
    Qwen 3.6 35B A3B — default model with 35B total parameters and 3B active per token.
    Uses hybrid sparse mixture-of-experts for efficient inference.
    Recommended for most use cases.
    """
    name: Literal["qwen/qwen3.6-35b-a3b"] = "qwen/qwen3.6-35b-a3b"
    value: Literal["qwen/qwen3.6-35b-a3b"] = "qwen/qwen3.6-35b-a3b"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Qwen 3.6 35B A3B"
        json_schema_extra = {"shortDescription": "Default, efficient inference"}


class Version27B(Config):
    """
    Qwen 3.6 27B — a mid-size model balancing performance and cost.
    Good for general vision-language tasks.
    """
    name: Literal["qwen/qwen3.6-27b"] = "qwen/qwen3.6-27b"
    value: Literal["qwen/qwen3.6-27b"] = "qwen/qwen3.6-27b"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Qwen 3.6 27B"
        json_schema_extra = {"shortDescription": "Mid-size, balanced"}


class VersionFlash(Config):
    """
    Qwen 3.6 Flash — the fastest and most cost-effective variant.
    Best for high-volume or latency-sensitive workloads.
    """
    name: Literal["qwen/qwen3.6-flash"] = "qwen/qwen3.6-flash"
    value: Literal["qwen/qwen3.6-flash"] = "qwen/qwen3.6-flash"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Qwen 3.6 Flash"
        json_schema_extra = {"shortDescription": "Fastest, lowest cost"}


class VersionPlus(Config):
    """
    Qwen 3.6 Plus — enhanced capability over the base models.
    Suitable for more demanding visual understanding tasks.
    """
    name: Literal["qwen/qwen3.6-plus"] = "qwen/qwen3.6-plus"
    value: Literal["qwen/qwen3.6-plus"] = "qwen/qwen3.6-plus"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Qwen 3.6 Plus"
        json_schema_extra = {"shortDescription": "Enhanced capability"}


class VersionMaxPreview(Config):
    """
    Qwen 3.6 Max Preview — the most capable model in the Qwen 3.6 family.
    Preview release; best for complex reasoning and visual analysis tasks.
    """
    name: Literal["qwen/qwen3.6-max-preview"] = "qwen/qwen3.6-max-preview"
    value: Literal["qwen/qwen3.6-max-preview"] = "qwen/qwen3.6-max-preview"
    type: Literal["string"] = "string"
    field: Literal["option"] = "option"

    class Config:
        title = "Qwen 3.6 Max Preview"
        json_schema_extra = {"shortDescription": "Most capable, preview"}


class InputModelVersion(Config):
    """
    Select the Qwen 3.6 model variant to use.
    Flash is fastest and cheapest. Max Preview is the most capable.
    35B A3B is the recommended default for most tasks.
    """
    name: Literal["inputModelVersion"] = "inputModelVersion"
    value: Union[Version35BA3B, Version27B, VersionFlash, VersionPlus, VersionMaxPreview]
    type: Literal["object"] = "object"
    field: Literal["dropdownlist"] = "dropdownlist"

    class Config:
        title = "Model Version"
        json_schema_extra = {"shortDescription": "Qwen Model"}


class TextPromptConfigs(Configs):
    inputPrompt: InputPrompt
    inputApiKey: InputOpenRouterApiKey
    inputModelVersion: InputModelVersion
    inputTemperature: TemperatureConfig
    maxTokens: MaxTokens
    maxConcurrentRequests: MaxConcurrentRequests


class TextPromptOutputs(Outputs):
    output: OutputText


class TextPromptRequest(Request):
    configs: TextPromptConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class TextPromptResponse(Response):
    outputs: TextPromptOutputs


class TextPrompt(Config):
    """
    Generates a text response based on a custom prompt with no image input.
    Use for pure text tasks such as content generation, Q&A, or reasoning.
    No image is required for this task.
    """
    name: Literal["TextPrompt"] = "TextPrompt"
    value: Union[TextPromptRequest, TextPromptResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Text Prompt"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Text-only generation"}


class UnconstrainedConfigs(Configs):
    inputPrompt: InputPrompt
    inputApiKey: InputOpenRouterApiKey
    inputModelVersion: InputModelVersion
    inputTemperature: TemperatureConfig
    maxTokens: MaxTokens
    maxConcurrentRequests: MaxConcurrentRequests


class UnconstrainedInputs(Inputs):
    inputImage: InputImage


class UnconstrainedOutputs(Outputs):
    output: OutputText


class UnconstrainedRequest(Request):
    inputs: Optional[UnconstrainedInputs]
    configs: UnconstrainedConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class UnconstrainedResponse(Response):
    outputs: UnconstrainedOutputs


class Unconstrained(Config):
    """
    Analyzes an image using a fully custom prompt without predefined constraints.
    Provides maximum flexibility for open-ended visual analysis.
    Use when no structured output format is required.
    """
    name: Literal["Unconstrained"] = "Unconstrained"
    value: Union[UnconstrainedRequest, UnconstrainedResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Open Prompt"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Free-form image analysis"}



class OCRConfigs(Configs):
    inputApiKey: InputOpenRouterApiKey
    inputModelVersion: InputModelVersion
    inputTemperature: TemperatureConfig
    maxTokens: MaxTokens
    maxConcurrentRequests: MaxConcurrentRequests


class OCRInputs(Inputs):
    inputImage: InputImage


class OCROutputs(Outputs):
    output: OutputText


class OCRRequest(Request):
    inputs: Optional[OCRInputs]
    configs: OCRConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class OCRResponse(Response):
    outputs: OCROutputs


class OCR(Config):
    """
    Extracts all text present in an image using optical character recognition.
    Returns the detected text as a plain string preserving paragraph structure.
    Works on documents, signs, labels, and handwritten content.
    """
    name: Literal["OCR"] = "OCR"
    value: Union[OCRRequest, OCRResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Text Recognition (OCR)"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Text extraction from images"}


class VQAConfigs(Configs):
    inputPrompt: InputPrompt
    inputApiKey: InputOpenRouterApiKey
    inputModelVersion: InputModelVersion
    inputTemperature: TemperatureConfig
    maxTokens: MaxTokens
    maxConcurrentRequests: MaxConcurrentRequests


class VQAInputs(Inputs):
    inputImage: InputImage


class VQAOutputs(Outputs):
    output: OutputText


class VQARequest(Request):
    inputs: Optional[VQAInputs]
    configs: VQAConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class VQAResponse(Response):
    outputs: VQAOutputs


class VisualQuestionAnswering(Config):
    """
    Answers a specific question about the content of an image.
    Provide a question as the prompt and receive a targeted answer.
    Useful for structured visual inspection and querying.
    """
    name: Literal["VisualQuestionAnswering"] = "VisualQuestionAnswering"
    value: Union[VQARequest, VQAResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Visual Question Answering"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Image question answering"}


class CaptionConfigs(Configs):
    inputApiKey: InputOpenRouterApiKey
    inputModelVersion: InputModelVersion
    inputTemperature: TemperatureConfig
    maxTokens: MaxTokens
    maxConcurrentRequests: MaxConcurrentRequests


class CaptionInputs(Inputs):
    inputImage: InputImage


class CaptionOutputs(Outputs):
    output: OutputText


class CaptionRequest(Request):
    inputs: Optional[CaptionInputs]
    configs: CaptionConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class CaptionResponse(Response):
    outputs: CaptionOutputs


class ShortCaption(Config):
    """
    Generates a concise one or two sentence caption describing an image.
    Suitable for labeling, quick summaries, or image metadata.
    Use Detailed Captioning for richer, longer descriptions.
    """
    name: Literal["ShortCaption"] = "ShortCaption"
    value: Union[CaptionRequest, CaptionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Captioning (Short)"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Brief image description"}



class DetailedCaptionConfigs(Configs):
    inputApiKey: InputOpenRouterApiKey
    inputModelVersion: InputModelVersion
    inputTemperature: TemperatureConfig
    maxTokens: MaxTokens
    maxConcurrentRequests: MaxConcurrentRequests


class DetailedCaptionInputs(Inputs):
    inputImage: InputImage


class DetailedCaptionOutputs(Outputs):
    output: OutputText


class DetailedCaptionRequest(Request):
    inputs: Optional[DetailedCaptionInputs]
    configs: DetailedCaptionConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class DetailedCaptionResponse(Response):
    outputs: DetailedCaptionOutputs


class DetailedCaption(Config):
    """
    Generates a comprehensive multi-sentence description of an image.
    Covers objects, attributes, spatial relationships, and scene context.
    Use for thorough image documentation or accessibility alt-text.
    """
    name: Literal["DetailedCaption"] = "DetailedCaption"
    value: Union[DetailedCaptionRequest, DetailedCaptionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Captioning (Detailed)"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Rich image description"}


class ClassificationConfigs(Configs):
    inputClasses: InputClasses
    inputApiKey: InputOpenRouterApiKey
    inputModelVersion: InputModelVersion
    inputTemperature: TemperatureConfig
    maxTokens: MaxTokens
    maxConcurrentRequests: MaxConcurrentRequests


class ClassificationInputs(Inputs):
    inputImage: InputImage


class ClassificationOutputs(Outputs):
    output: OutputText
    classes: Classes


class ClassificationRequest(Request):
    inputs: Optional[ClassificationInputs]
    configs: ClassificationConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class ClassificationResponse(Response):
    outputs: ClassificationOutputs


class Classification(Config):
    """
    Assigns exactly one class label from a predefined list to an image.
    The model selects the single most appropriate class.
    Provide the list of valid classes using the Classes config.
    """
    name: Literal["Classification"] = "Classification"
    value: Union[ClassificationRequest, ClassificationResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Single-Label Classification"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Single class assignment"}



class MultiLabelConfigs(Configs):
    inputClasses: InputClasses
    inputApiKey: InputOpenRouterApiKey
    inputModelVersion: InputModelVersion
    inputTemperature: TemperatureConfig
    maxTokens: MaxTokens
    maxConcurrentRequests: MaxConcurrentRequests


class MultiLabelInputs(Inputs):
    inputImage: InputImage


class MultiLabelOutputs(Outputs):
    output: OutputText
    classes: Classes


class MultiLabelRequest(Request):
    inputs: Optional[MultiLabelInputs]
    configs: MultiLabelConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class MultiLabelResponse(Response):
    outputs: MultiLabelOutputs


class MultiLabel(Config):
    """
    Assigns one or more class labels from a predefined list to an image.
    The model selects all classes that apply to the image.
    Provide the list of valid classes using the Classes config.
    """
    name: Literal["MultiLabel"] = "MultiLabel"
    value: Union[MultiLabelRequest, MultiLabelResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Multi-Label Classification"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Multiple class assignment"}


class ObjectDetectionConfigs(Configs):
    inputClasses: InputClasses
    inputApiKey: InputOpenRouterApiKey
    inputModelVersion: InputModelVersion
    inputTemperature: TemperatureConfig
    maxTokens: MaxTokens
    maxConcurrentRequests: MaxConcurrentRequests


class ObjectDetectionInputs(Inputs):
    inputImage: InputImage


class ObjectDetectionOutputs(Outputs):
    output: OutputText
    classes: Classes


class ObjectDetectionRequest(Request):
    inputs: Optional[ObjectDetectionInputs]
    configs: ObjectDetectionConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class ObjectDetectionResponse(Response):
    outputs: ObjectDetectionOutputs


class ObjectDetection(Config):
    """
    Detects and identifies objects in an image from a predefined class list.
    Returns detected class names found within the image.
    Provide the list of target classes using the Classes config.
    """
    name: Literal["ObjectDetection"] = "ObjectDetection"
    value: Union[ObjectDetectionRequest, ObjectDetectionResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Object Detection"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Detect objects by class"}


class StructuredAnsweringConfigs(Configs):
    inputOutputStructure: InputOutputStructure
    inputApiKey: InputOpenRouterApiKey
    inputModelVersion: InputModelVersion
    inputTemperature: TemperatureConfig
    maxTokens: MaxTokens
    maxConcurrentRequests: MaxConcurrentRequests


class StructuredAnsweringInputs(Inputs):
    inputImage: InputImage


class StructuredAnsweringOutputs(Outputs):
    output: OutputText


class StructuredAnsweringRequest(Request):
    inputs: Optional[StructuredAnsweringInputs]
    configs: StructuredAnsweringConfigs

    class Config:
        json_schema_extra = {"target": "configs"}


class StructuredAnsweringResponse(Response):
    outputs: StructuredAnsweringOutputs


class StructuredAnswering(Config):
    """
    Generates a structured JSON output based on a user-defined schema and image.
    Define the expected fields and their descriptions in the Output Structure config.
    Ideal for extracting structured data from visual content.
    """
    name: Literal["StructuredAnswering"] = "StructuredAnswering"
    value: Union[StructuredAnsweringRequest, StructuredAnsweringResponse]
    type: Literal["object"] = "object"
    field: Literal["option"] = "option"

    class Config:
        title = "Structured Output Generation"
        json_schema_extra = {"target": {"value": 0}, "shortDescription": "Structured data extraction"}


class ConfigExecutor(Config):
    """
    Select the vision task to perform on the input image.
    Each task has its own prompt, output format, and configuration options.
    Choose the task that best matches your use case.
    """
    name: Literal["ConfigExecutor"] = "ConfigExecutor"
    value: Union[
        TextPrompt,
        Unconstrained,
        OCR,
        VisualQuestionAnswering,
        ShortCaption,
        DetailedCaption,
        Classification,
        MultiLabel,
        ObjectDetection,
        StructuredAnswering,
    ]
    type: Literal["executor"] = "executor"
    field: Literal["dependentDropdownlist"] = "dependentDropdownlist"

    class Config:
        title = "Task"
        json_schema_extra = {"shortDescription": "Select Vision Task"}


class PackageConfigs(Configs):
    executor: ConfigExecutor


class PackageModel(Package):
    name: Literal["QwenOpenRouter"] = "QwenOpenRouter"
    configs: PackageConfigs
    type: Literal["capsule"] = "capsule"