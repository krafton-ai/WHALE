# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize tools from config file
        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        max_assistant_tokens = self.rollout_config.multi_turn.max_assistant_tokens
        self.max_assistant_tokens = (
            int(max_assistant_tokens) if max_assistant_tokens else None
        )
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side
        tool_config_path = self.rollout_config.multi_turn.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # Initialize interactions from config file
        self.interaction_config_file = self.rollout_config.multi_turn.interaction_config_path
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(
                self.interaction_config_file
            )

    def _assistant_tokens_used(self, agent_data: AgentData) -> int:
        return int(sum(1 for mask in agent_data.response_mask if mask))

    def _remaining_assistant_tokens(self, agent_data: AgentData) -> int | None:
        if not self.max_assistant_tokens or self.max_assistant_tokens <= 0:
            return None
        return max(0, int(self.max_assistant_tokens) - self._assistant_tokens_used(agent_data))

    def _cap_sampling_to_remaining_assistant_tokens(
        self, agent_data: AgentData, sampling_params: dict[str, Any]
    ) -> dict[str, Any] | None:
        remaining_total = self.response_length - len(agent_data.response_mask)
        if remaining_total <= 0:
            return None

        remaining_assistant = self._remaining_assistant_tokens(agent_data)
        if remaining_assistant is None:
            return sampling_params
        if remaining_assistant <= 0:
            return None

        capped = dict(sampling_params)
        per_call_cap = min(remaining_assistant, remaining_total)
        requested = capped.get("max_tokens")
        if requested is None:
            requested = capped.pop("max_new_tokens", None)
        if requested is not None:
            try:
                per_call_cap = min(per_call_cap, int(requested))
            except (TypeError, ValueError):
                logger.warning("Ignoring non-integer max_tokens=%r", requested)
        if per_call_cap <= 0:
            return None
        capped["max_tokens"] = per_call_cap
        return capped

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        # Meta-harness hook: when HARNESS_PATH points to a candidate that
        # OVERRIDES setup_state (e.g. h29 axis-E pre-fetch which mutates the
        # initial user message before turn 1), dispatch it BEFORE
        # apply_chat_template so the mutation propagates into the tokenized
        # prompt. No-op for HARNESS_PATH unset, base alias, candidates that
        # inherit setup_state without overriding (e.g. h21), or trajectories
        # where setup_state has already run. See
        # verl/utils/meta_harness_hook.py:run_candidate_setup_state_if_needed.
        try:
            from verl.utils.meta_harness_hook import (
                run_candidate_setup_state_if_needed,
            )

            await run_candidate_setup_state_if_needed(agent_data)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "meta-harness pending-state setup_state dispatch skipped: %s", exc
            )

        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=self.tool_schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []
        turn_sampling_params = self._cap_sampling_to_remaining_assistant_tokens(
            agent_data, sampling_params
        )
        if turn_sampling_params is None and not ignore_termination:
            return AgentState.TERMINATED

        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=turn_sampling_params or sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            # Multi-round calls, only update the maximum max_global_steps.
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        agent_data.metrics["assistant_tokens"] = self._assistant_tokens_used(agent_data)
        if self.max_assistant_tokens:
            agent_data.metrics["assistant_token_budget"] = self.max_assistant_tokens
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        remaining_assistant = self._remaining_assistant_tokens(agent_data)
        if not ignore_termination and remaining_assistant is not None and remaining_assistant <= 0:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Meta-harness hook: when HARNESS_PATH points to a CandidateEnv, run any
        # candidate-defined @vf.stop hooks (other than no_tools_called /
        # max_turns_reached, dispatched elsewhere) so custom early-stop
        # conditions terminate the rollout exactly as they do during
        # meta-harness eval. No-op if HARNESS_PATH is unset. See
        # verl/utils/meta_harness_hook.py:check_candidate_extra_stop_hooks.
        try:
            from verl.utils.meta_harness_hook import check_candidate_extra_stop_hooks

            if await check_candidate_extra_stop_hooks(agent_data):
                return AgentState.TERMINATED
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("meta-harness extra stop-hook dispatch skipped: %s", exc)

        # Extract tool calls
        tools = [tool.tool_schema for tool in self.tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING

        # No tool calls. verl's default is TERMINATED. The meta-harness hook
        # can override this when HARNESS_PATH is set and the candidate's
        # `no_tools_called` (a `@vf.stop` predicate inherited from verifiers)
        # has been redefined relative to base SearchR1Env. The override may
        # return False (= keep generating), in which case we mirror eval-side
        # `SearchR1Env.get_prompt_messages` by appending the same nudge user
        # message and returning AgentState.GENERATING for one more turn.
        # See verl/utils/meta_harness_hook.py:should_terminate_on_no_tool_calls.
        try:
            from verl.utils.meta_harness_hook import (
                should_terminate_on_no_tool_calls,
            )

            # Supply a lazy decoder so the hook can synthesize the
            # just-generated assistant turn into state["trajectory"] before
            # calling candidate.no_tools_called / get_prompt_messages. Verl
            # never appends the assistant message to agent_data.messages on
            # the no-interaction-config path (which is Search-R1's normal
            # mode), so without this the hook's reconstruction returned []
            # and h21-style gate/nudge logic was silently bypassed.
            async def _decode_assistant():
                return await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.decode(
                        agent_data.response_ids, skip_special_tokens=True
                    ),
                )

            should_stop, nudge_messages = await should_terminate_on_no_tool_calls(
                agent_data,
                decode_assistant=_decode_assistant,
            )
            if not should_stop and nudge_messages:
                # Append the candidate-driven nudge as additional user-role
                # context, identical structure to the tool-response append
                # path used by _handle_processing_tools_state_candidate.
                agent_data.messages.extend(nudge_messages)
                response_ids = await self.apply_chat_template(
                    nudge_messages,
                    images=None,
                    videos=None,
                    remove_system_prompt=True,
                )
                if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
                    return AgentState.TERMINATED
                agent_data.prompt_ids += response_ids
                agent_data.response_mask += [0] * len(response_ids)
                if agent_data.response_logprobs:
                    agent_data.response_logprobs += [0.0] * len(response_ids)
                agent_data.user_turns += 1
                return AgentState.GENERATING
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "meta-harness no_tools_called dispatch skipped: %s", exc
            )

        return AgentState.TERMINATED

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        # Meta-harness hook: when HARNESS_PATH points to a CandidateEnv, route the
        # tool-execution + tool_response assembly through candidate.env_response so
        # the harness chosen during meta-harness eval is reproduced inside training.
        # No-op if HARNESS_PATH is unset. See verl/utils/meta_harness_hook.py.
        try:
            from verl.utils.meta_harness_hook import (
                call_candidate_env_response,
                should_dispatch_to_candidate,
            )

            if should_dispatch_to_candidate():
                return await self._handle_processing_tools_state_candidate(
                    agent_data, call_candidate_env_response
                )
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("meta-harness env_response dispatch skipped: %s", exc)

        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        # Track per-trajectory total tool-call count. simple_timer below already
        # accumulates wall time under "tool_calls"; this counter accumulates the
        # call count across all assistant turns of this trajectory so that
        # agent_loop._performance_metrics can emit num_tool_calls/{min,max,mean}.
        agent_data.metrics["num_tool_calls"] = (
            agent_data.metrics.get("num_tool_calls", 0) + len(tasks)
        )

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        for tool_response, tool_reward, _ in responses:
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_processing_tools_state_candidate(
        self, agent_data: AgentData, call_candidate_env_response
    ) -> AgentState:
        """Meta-harness env_response path: call candidate.env_response instead of
        running verl's per-tool execution loop.

        Replaces the entire body of _handle_processing_tools_state for the
        HARNESS_PATH-set case. Tool execution happens INSIDE candidate.env_response
        (which calls candidate.TOOLS[i] directly, bypassing verl SearchTool).

        On candidate exception: returns AgentState.TERMINATED to drop the
        trajectory (matches verl's malformed-tool-call policy per Q3).
        """
        agent_data.metrics["num_tool_calls"] = (
            agent_data.metrics.get("num_tool_calls", 0) + len(agent_data.tool_calls)
        )

        async def _decode_assistant():
            return await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True),
            )

        with simple_timer("tool_calls", agent_data.metrics):
            # Forward verl's assistant-turn cap so candidate.max_turns matches
            # what meta-harness eval's load_environment(max_turns=...) sets.
            add_messages = await call_candidate_env_response(
                agent_data,
                _decode_assistant,
                max_turns=self.max_assistant_turns,
            )

        if add_messages is None:
            # candidate.env_response raised — drop trajectory.
            return AgentState.TERMINATED

        agent_data.messages.extend(add_messages)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_call_names = [m.get("role", "tool") for m in add_messages]
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            response_ids = await self.apply_chat_template(
                add_messages,
                images=None,
                videos=None,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_name = tool_call.name
            tool_args = json.loads(tool_call.arguments)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    def _initialize_interactions(self, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        return interaction_map
