from copy import deepcopy
import json
from typing import List, Optional, Callable
import logging
import openai
from .tools import gather_tools

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ToolFuser:

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        openai_client: openai.OpenAI = None,
        model: str = "qwen3-30b-a3b",
        stream_handler: Optional[Callable] = None,
        debug: bool = False,
    ):
        self.client = openai_client or openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.stream_handler = stream_handler
        self.chat_history = []
        self.debug = debug
        if self.debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    def _execute_tool(
        self,
        tool: Callable,
        assistant_content: str,
        tool_call_id: str,
        tool_name: str,
        tool_args: dict,
        context: Optional[dict] = None,
    ) -> str:
        tool_messages = [
            {
                "role": "assistant",
                "content": assistant_content or None,
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": None,
            },
        ]
        try:
            if context:
                tool_args["context"] = context

            result = tool(**tool_args)
        except Exception as e:
            result = f"Error executing tool {tool_name}: {str(e)}"

        tool_messages[-1]["content"] = json.dumps(result)
        return tool_messages

    def invoke_chat_stream(
        self,
        inputs: List[dict],
        tools: Optional[List[Callable]] = None,
        temperature: float = 0.1,
        reasoning_effort: str = "minimal",
        system_prompt: Optional[str] = None,
        context: Optional[dict] = None,
    ):

        messages = deepcopy(inputs)
        if system_prompt and messages and messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_prompt})

        tool_metas, tools_mapping = gather_tools(tools) if tools else ([], {})
        tools_params = (
            {"tool_choice": "auto", "tools": tool_metas} if tool_metas else {}
        )

        if self.debug:
            logger.debug("=== Debug Info ===")
            logger.debug("Messages:", json.dumps(messages, indent=2))
            logger.debug("Tools:", json.dumps(tool_metas, indent=2))
            logger.debug("==================")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            **tools_params,
        )

        assistant_content = ""
        tool_call_id = None
        tool_name = None
        tool_args_json = ""

        for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if not delta:
                continue
            # Text tokens
            if getattr(delta, "content", None):
                if self.stream_handler:
                    self.stream_handler(delta.content)
                assistant_content += delta.content

            # Tool call deltas
            tcs = getattr(delta, "tool_calls", None)
            if tcs:
                call = tcs[0]
                tool_call_id = call.id or tool_call_id
                if call.function:
                    if call.function.name:
                        tool_name = call.function.name
                    if call.function.arguments:
                        tool_args_json += call.function.arguments

        if tool_call_id and tool_name:
            args = json.loads(tool_args_json or "{}")
            fuct, stateless = tools_mapping.get(tool_name)
            ctx = context if not stateless else None
            tool_messages = self._execute_tool(
                fuct, assistant_content, tool_call_id, tool_name, args, ctx
            )
            messages.extend(tool_messages)
            return self.invoke_chat_stream(
                messages,
                tools=tools,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                context=ctx,
            )
        return assistant_content

    def invoke_chat(
        self,
        inputs: List[dict],
        tools: Optional[List[Callable]] = None,
        temperature: float = 0.1,
        reasoning_effort: str = "minimal",
        system_prompt: Optional[str] = None,
        context: Optional[dict] = None,
    ):
        messages = deepcopy(inputs)
        if system_prompt and messages and messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": system_prompt})

        tool_metas, tools_mapping = gather_tools(tools) if tools else ([], {})
        tools_params = (
            {"tool_choice": "auto", "tools": tool_metas} if tool_metas else {}
        )

        if self.debug:
            logger.debug("=== Debug Info ===")
            logger.debug("Messages:", json.dumps(messages, indent=2))
            logger.debug("Tools:", json.dumps(tool_metas, indent=2))
            logger.debug("==================")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            **tools_params,
        )

        assistant_content = ""
        tool_call_id = None
        tool_name = None
        tool_args_json = ""
        logger.info("Response:", response.choices[0].finish_reason)
        if response.choices[0].finish_reason == "stop":
            assistant_content = response.choices[0].message.content

        tcs = getattr(response.choices[0].message, "tool_calls", None)
        if tcs:
            call = tcs[0]
            tool_call_id = call.id or tool_call_id
            if call.function:
                if call.function.name:
                    tool_name = call.function.name
                if call.function.arguments:
                    tool_args_json += call.function.arguments
        if tool_call_id and tool_name:
            args = json.loads(tool_args_json or "{}")
            fuct, stateless = tools_mapping.get(tool_name)
            ctx = context if not stateless else None
            tool_messages = self._execute_tool(
                fuct, assistant_content, tool_call_id, tool_name, args, ctx
            )
            messages.extend(tool_messages)
            return self.invoke_chat(
                messages,
                tools=tools,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                context=ctx,
            )

        return assistant_content
