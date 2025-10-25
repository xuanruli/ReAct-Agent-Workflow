from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Literal, Type, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_core.runnables import RunnableGenerator
from langgraph.types import Interrupt, StreamMode
from pydantic import BaseModel, Field

EventType = Literal[
    "message",
    "message_start",
    "message_end",
    "tool_call",
    "tool_call_output",
    "tool_call_start",
    "tool_call_end",
    "tool_oauth",
]


class StreamEvent(BaseModel):
    event_type: EventType = Field(description="The type of the event")


class StreamMessage(StreamEvent):
    event_type: Literal["message"] = "message"

    id: str = Field(description="The id of the message")
    type: Literal["text", "thinking"] = Field(description="The type of the message")
    content: str = Field(description="The content of the message")


class ToolCall(StreamEvent):
    event_type: Literal["tool_call"] = "tool_call"

    id: str = Field(description="The id of the tool call")
    name: str = Field(description="The name of the tool")
    args: dict[str, Any] = Field(description="The arguments of the tool call")


class ToolCallOutput(StreamEvent):
    event_type: Literal["tool_call_output"] = "tool_call_output"

    id: str = Field(description="The id of the tool call")
    name: str = Field(description="The name of the tool")
    output: str = Field(description="The output of the tool call")
    missing: bool = Field(default=False, description="Whether the tool call is missing")


class MessageStart(StreamEvent):
    event_type: Literal["message_start"] = "message_start"

    id: str = Field(description="The id of the message")
    type: Literal["text", "thinking"] = Field(description="The type of the message")
    timestamp: float = Field(description="The timestamp of the message")


class MessageEnd(StreamEvent):
    event_type: Literal["message_end"] = "message_end"

    id: str = Field(description="The id of the message")
    content: str = Field(description="The full content of the message")


class ToolCallStart(StreamEvent):
    event_type: Literal["tool_call_start"] = "tool_call_start"

    id: str = Field(description="The id of the tool call")
    name: str = Field(description="The name of the tool")
    timestamp: float = Field(description="The timestamp of the tool call")


class ToolCallEnd(StreamEvent):
    event_type: Literal["tool_call_end"] = "tool_call_end"

    id: str = Field(description="The id of the tool call")


class ToolOAuth(StreamEvent):
    event_type: Literal["tool_oauth"] = "tool_oauth"

    id: str = Field(description="The id of the tool call")
    name: str = Field(description="The name of the tool")
    url: str = Field(description="The url of the oauth")

    @classmethod
    def from_interrupt(cls, id: str, name: str, interrupt: Interrupt) -> ToolOAuth:
        url = interrupt.value.split("Please use the following link to authorize: ")[1]
        return cls(id=id, name=name, url=url)

    @classmethod
    def from_tool_call_output(cls, id: str, name: str, output: Any) -> ToolOAuth:
        try:
            json_output = json.loads(output)
            error = json_output["error"]
            url = error.split("Please use the following link to authorize: ")[1]
            return cls(id=id, name=name, url=url)
        except Exception as e:
            print(f"Error parsing tool call output: {e}")
            return cls(id=id, name=name, url="")


StreamPart = (
    StreamMessage | ToolCall | ToolCallOutput | MessageStart | MessageEnd | ToolCallStart | ToolCallEnd | ToolOAuth
)


INTERRUPT_MARKER = "__interrupt__"
TOOL_STATE = "tools"
AGENT_STATE = "agent"


class _ToolCallDelta(BaseModel):
    id: str = Field(description="The id of the tool call")
    name: str = Field(description="The name of the tool")
    args: str = Field(default="", description="The delta of the arguments of the tool call")

    def is_valid(self) -> bool:
        try:
            json.loads(self.args)
            return True
        except json.JSONDecodeError:
            return False

    def to_tool_call(self) -> ToolCall:
        return ToolCall(
            id=self.id,
            name=self.name,
            args=json.loads(self.args),
        )


def _random_id() -> str:
    return str(uuid.uuid4())


async def _parse_openai_agent_output(
    agent_stream: AsyncIterator[tuple[StreamMode, Any]],
) -> AsyncIterator[StreamPart]:
    tool_call_delta: _ToolCallDelta | None = None
    last_content_type: str | None = None
    current_message_id: str | None = None
    current_message_type: Literal["text", "thinking"] | None = None

    async for stream_type, part in agent_stream:
        if stream_type == "messages":
            chunk, _ = cast(tuple[Any, Any], part)
            if not isinstance(chunk, AIMessageChunk):
                continue

            if len(chunk.content) == 0:
                continue

            if len(chunk.content) > 1:
                print(chunk.content)

            content = cast(dict[str, Any], chunk.content[0])

            if "type" not in content:
                continue

            content_type: str = content["type"]

            tool_call_finished = last_content_type == "function_call" and last_content_type != content_type

            new_tool_call_started = content_type == "function_call" and "name" in content and "call_id" in content

            if (tool_call_finished or new_tool_call_started) and tool_call_delta is not None:
                yield tool_call_delta.to_tool_call()
                tool_call_delta = None

            last_content_type = content_type
            if isinstance(content, str) or content_type == "text":
                # Check if we need a new message ID
                if current_message_type != "text":
                    current_message_id = _random_id()
                    current_message_type = "text"

                assert current_message_id is not None, "current_message_id is None"
                yield StreamMessage(id=current_message_id, type="text", content=content["text"])
            elif content_type == "reasoning":
                # Check if we need a new message ID for thinking
                if current_message_type != "thinking":
                    current_message_id = _random_id()
                    current_message_type = "thinking"

                assert current_message_id is not None, "current_message_id is None"
                summary = content.get("summary", [])
                for item in summary:
                    assert item["type"] == "summary_text"

                    yield StreamMessage(id=current_message_id, type="thinking", content=item["text"])
            elif content_type == "function_call":
                # Reset message tracking when we start a function call
                current_message_type = None
                current_message_id = None
                if "name" in content and "call_id" in content:
                    # indicate start of tool call
                    tool_call_delta = _ToolCallDelta(
                        id=content["call_id"],
                        name=content["name"],
                    )

                if "arguments" in content:
                    # indicate streams of arguments
                    assert tool_call_delta is not None
                    tool_call_delta.args += content["arguments"]
            else:
                raise ValueError(f"Unknown content type: {content}")
        elif stream_type == "updates":
            part = cast(dict[str, Any], part)

            if tool_call_delta and tool_call_delta.is_valid():
                yield tool_call_delta.to_tool_call()
                tool_call_delta = None

            # Reset message tracking on updates
            current_message_type = None
            current_message_id = None
            last_content_type = None

            for message in part.get(TOOL_STATE, {}).get("messages", []):
                assert isinstance(message, ToolMessage)
                assert message.name is not None, "message.name is None"

                yield ToolCallOutput(
                    id=message.tool_call_id,
                    name=message.name,
                    output=str(message.content),
                )

    if tool_call_delta and tool_call_delta.is_valid():
        yield tool_call_delta.to_tool_call()


async def _react_stream_postprocess_parser(
    agent_stream: AsyncIterator[StreamPart],
) -> AsyncIterator[StreamPart]:
    pending_tool_calls: dict[str, ToolCall] = {}
    last_part: StreamPart | None = None
    last_message_type: Type[StreamPart] | None = None
    last_message_id: str | None = None
    accumulated_message_content: str = ""

    async for part in agent_stream:
        curr_message_type: Type[StreamPart] = type(part)

        if curr_message_type != last_message_type or (
            isinstance(part, StreamMessage) and isinstance(last_part, StreamMessage) and part.type != last_part.type
        ):
            if last_message_type is not None and last_message_type == StreamMessage:
                assert last_message_id is not None
                yield MessageEnd(id=last_message_id, content=accumulated_message_content)
            if curr_message_type == StreamMessage:
                assert isinstance(part, StreamMessage)
                yield MessageStart(id=part.id, type=part.type, timestamp=time.time())
                last_message_id = part.id

            # reset accumulated message content
            accumulated_message_content = ""

        last_part = part
        last_message_type = curr_message_type

        if isinstance(part, StreamMessage):
            accumulated_message_content += part.content
            yield part
        elif isinstance(part, ToolCall):
            pending_tool_calls[part.id] = part
            yield ToolCallStart(id=part.id, name=part.name, timestamp=time.time())
            yield part
        elif isinstance(part, ToolCallOutput):
            if part.id not in pending_tool_calls:
                raise ValueError(f"Tool call output for unknown tool call: {part.id}")

            tool_call = pending_tool_calls.pop(part.id)
            yield part
            yield ToolCallEnd(id=tool_call.id)
        elif isinstance(part, ToolOAuth):
            yield part
        else:
            raise ValueError(f"Unknown part type: {type(part)}")

    if last_message_type is not None and last_message_type == StreamMessage:
        assert last_message_id is not None
        yield MessageEnd(id=last_message_id, content=accumulated_message_content)

    for tool_call in pending_tool_calls.values():
        yield ToolCallOutput(id=tool_call.id, name=tool_call.name, output="", missing=True)
        yield ToolCallEnd(id=tool_call.id)


async def _pretty_stream_react_agent(
    agent_stream: AsyncIterator[StreamPart],
) -> AsyncIterator[str]:
    last_message_type: Literal["text", "thinking"] | None = None

    async for part in agent_stream:
        if isinstance(part, MessageStart):
            last_message_type = part.type
            yield f"<{part.type} id={part.id} timestamp={part.timestamp}>\n"
        elif isinstance(part, MessageEnd):
            yield f"\n</{last_message_type}>\n"
            yield f'<message id={part.id} content="{part.content.replace("\n", " ")}">\n'
            last_message_type = None
        elif isinstance(part, StreamMessage):
            yield part.content
        elif isinstance(part, ToolCall):
            yield f"{part.args}\n</tool_call>\n"
        elif isinstance(part, ToolCallOutput):
            yield (
                f"<tool_call_output name={part.name} id={part.id}>"
                f"{part.output if not part.missing else '[missing tool call output]'}</tool_call_output>\n"
            )
        elif isinstance(part, ToolOAuth):
            yield f"<oauth id={part.id} name={part.name}>\n{part.url}\n</oauth>\n"
        elif isinstance(part, ToolCallStart):
            yield f"<tool_call name={part.name} id={part.id} timestamp={part.timestamp}>\n"


def create_react_agent_parser(
    model: BaseChatModel,
):
    postprocess_parser = RunnableGenerator(_react_stream_postprocess_parser)
    return RunnableGenerator(_parse_openai_agent_output) | postprocess_parser


pretty_stream_react_agent = RunnableGenerator(_pretty_stream_react_agent)
