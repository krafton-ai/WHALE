# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# `extra="allow"` so that JSON-Schema fields not declared below (e.g. items,
# default, title, minimum/maximum, minItems/maxItems, additionalProperties) are
# preserved through `model_dump(exclude_unset=True, exclude_none=True)` instead
# of being silently dropped. This matters because verl's tool_agent_loop
# embeds the dumped schema verbatim in the `<tools>` block of the system
# prompt, and any field stripped here disappears from what the model sees —
# which prevents bit-identical alignment with the meta-harness verifiers
# stack (where convert_func_to_tool_def emits a richer schema).
#
# Field declarations are pruned to the minimum needed to (a) keep the
# attribute-access surface other verl modules rely on (`.function.name`,
# `.function.parameters.properties`) and (b) match the *key order* that
# pydantic-v2 produces for `model_json_schema()`-style dicts (which is what
# verifiers emits). Pydantic dumps declared fields first in declaration
# order, then extras in input-dict insertion order; by declaring only the
# fields that appear first in the verifiers output and pushing everything
# else through extras, the YAML author controls the trailing key order.
#
#   verifiers per-property order: default, description, items, title, type
#     → declare nothing; let YAML order win.
#   verifiers parameters order:   properties, required, title, type, additionalProperties
#     → declare `properties, required`; YAML adds title/type/additionalProperties
#       in that order as extras.
#   verifiers function order:     name, description, parameters
#     → declare all three.
#   verifiers tool order:         type, function
#     → declare both.
class OpenAIFunctionPropertySchema(BaseModel):
    """The schema of a parameter in OpenAI format.

    No declared fields — all properties (type, description, default, items,
    title, enum, minimum/maximum, etc.) flow through `extras` so the YAML
    author controls dump key order. Attribute access (`prop.type`,
    `prop.description`, `prop.default`) still works via Pydantic's
    `__pydantic_extra__` descriptor.
    """

    model_config = ConfigDict(extra="allow")


class OpenAIFunctionParametersSchema(BaseModel):
    """The schema of parameters in OpenAI format.

    Declares `properties, required` so they appear first in dumps and so that
    `properties` values are auto-validated as `OpenAIFunctionPropertySchema`
    (preserving attribute access through the chain). Everything else (`type`,
    `title`, `additionalProperties`, etc.) flows through extras in YAML
    insertion order.
    """

    model_config = ConfigDict(extra="allow")

    properties: dict[str, OpenAIFunctionPropertySchema] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class OpenAIFunctionSchema(BaseModel):
    """The schema of a function in OpenAI format."""

    model_config = ConfigDict(extra="allow")

    name: str
    description: str
    parameters: OpenAIFunctionParametersSchema = Field(
        default_factory=lambda: OpenAIFunctionParametersSchema(properties={}, required=[])
    )
    strict: bool = False


class OpenAIFunctionToolSchema(BaseModel):
    """The schema of a tool in OpenAI format."""

    model_config = ConfigDict(extra="allow")

    type: str
    function: OpenAIFunctionSchema


class OpenAIFunctionParsedSchema(BaseModel):
    """The parsed schema of a tool in OpenAI format."""

    name: str
    arguments: str  # JSON string


class OpenAIFunctionCallSchema(BaseModel):
    """The parsed schema of a tool in OpenAI format."""

    name: str
    arguments: dict[str, Any]

    @staticmethod
    def from_openai_function_parsed_schema(
        parsed_schema: OpenAIFunctionParsedSchema,
    ) -> tuple["OpenAIFunctionCallSchema", bool]:
        has_decode_error = False
        try:
            arguments = json.loads(parsed_schema.arguments)
        except json.JSONDecodeError:
            arguments = {}
            has_decode_error = True
        # If the arguments is not a dict, it means the arguments is not a valid JSON string
        if not isinstance(arguments, dict):
            arguments = {}
            has_decode_error = True

        return OpenAIFunctionCallSchema(name=parsed_schema.name, arguments=arguments), has_decode_error


class OpenAIFunctionToolCall(BaseModel):
    """The tool call in OpenAI format."""

    id: str
    type: Literal["function"] = "function"
    function: OpenAIFunctionCallSchema


class ToolResponse(BaseModel):
    """The response from a tool execution."""

    text: str | None = None
    image: list[Any] | None = None
    video: list[Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def initialize_request(cls, values):
        if "image" in values and not isinstance(values["image"], list):
            raise ValueError(
                f"Image must be a list, but got {type(values['image'])}. Please check the tool.execute(). "
                f"For single images, wrap in a list: [image]. "
                f"Example: {{'image': [img1]}} or {{'image': [img1, img2, ...]}}."
            )
        if "video" in values and not isinstance(values["video"], list):
            raise ValueError(
                f"Video must be a list, but got {type(values['video'])}. Please check the tool.execute(). "
                f"For single videos, wrap in a list: [video]. "
                f"Example: {{'video': [video1]}} or {{'video': [video1, video2, ...]}}."
            )

        return values

    def is_empty(self) -> bool:
        return not self.text and not self.image and not self.video

    def is_text_only(self) -> bool:
        return self.text and not self.image and not self.video
