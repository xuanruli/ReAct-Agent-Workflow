import inspect
from functools import wraps
from typing import List, Optional, Type

from langchain_core.messages import ToolMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import InjectedToolCallId, tool, StructuredTool
from langgraph.prebuilt import InjectedState
from react_types import State, StrategyItem
from langgraph.prebuilt.chat_agent_executor import AgentStatePydantic
from langgraph.types import Command
from prompt import HOOK_SYSTEM_PROMPT, NARRATIVE_SYSTEM_PROMPT
from pydantic import Field, ValidationError
from react_types import (
    AddHookStrategy,
    AddNarrativeStrategy,
    BaseStrategy,
    DeleteStrategy,
    HookStrategy,
    MergeHookStrategy,
    MergeNarrativeStrategy,
    NarrativeStrategy,
    RemoveStrategy,
    SplitHookStrategy,
    SplitNarrativeStrategy,
    StrategyItem,
    ToolType,
    UpdateHookStrategy,
    UpdateNarrativeStrategy,
)
from typing_extensions import Annotated
from utils.window_utils import VideoItemWithDeliveryStyle, VideoItemWithHook



def create_strategy_tool(strategy_type: type[ToolType]) -> StructuredTool:

    def wrapper(**kwargs):
        st = strategy_type.model_validate(kwargs)
        return Command(
            update={
                "strategy_set": st.mutate(st.state.strategy_set),
                "messages": [ToolMessage("Success", tool_call_id=st.tool_call_id)]
            }
        )

    return StructuredTool.from_function(
        name=strategy_type.__name__,
        description=strategy_type.__doc__, 
        func=wrapper,
        args_schema=strategy_type,
    )


def general_mutate_strategy(strategy_type: Type[ToolType]):
    def decorator(func):  # passing params
        @wraps(func)
        def wrapper(
            state: Annotated[State, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId], *args, **kwargs
        ):
            sig = inspect.signature(func)
            bound_args = sig.bind(state, tool_call_id, *args, **kwargs)
            bound_args.apply_defaults()

            # Remove state and tool_call_id from the arguments
            strategy_kwargs = {
                k: v for k, v in bound_args.arguments.items() if k not in ["state", "tool_call_id"]
            }
            
            required_fields = {name for name, field in strategy_type.model_fields.items() if field.is_required()}
            provided_fields = set(strategy_kwargs.keys())
            missing_fields = required_fields - provided_fields

            assert not missing_fields, f"Required fields {missing_fields} are missing"

            try:
                updated_strategy = strategy_type(**strategy_kwargs)
            except ValidationError as e:
                return Command(
                    update={
                        "messages": [ToolMessage(f"Validation error: {e}, strategy keywords: {strategy_kwargs}, Bound argument: {bound_args.arguments}", tool_call_id=tool_call_id)],
                    }
                )

            assert hasattr(updated_strategy, "mutate"), "Updated strategy tool must have a mutate method"

            return Command(
                update={
                    "strategy_set": updated_strategy.mutate(state.strategy_set),
                    "messages": [ToolMessage("Success", tool_call_id=tool_call_id)],
                }
            )

        return wrapper

    return decorator


@tool
@general_mutate_strategy(AddNarrativeStrategy)
def add_narrative_strategy(
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    strategy_name: Annotated[
        str,
        "Confidential Content",
    ],
    strategy_description: Annotated[
        str,
        "Confidential Content",
    ],
    structure_flow_description: Annotated[
        str,
        "Confidential Content",
    ],
):
    """Confidential Content"""

    pass


@tool
@general_mutate_strategy(AddHookStrategy)
def add_hook_strategy(
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    strategy_name: Annotated[
        str, "Confidential Content"
    ],
    strategy_description: Annotated[
        str, "Confidential Content"
    ],
    hook_selection_formula: Annotated[
        str, "Confidential Content"
    ],
    audience_emotion: Annotated[
        List[str],
        "Confidential Content"
    ],
    pacing_structure: Annotated[
        List[str],
        "Confidential Content"
    ],
    visual_elements: Annotated[
        List[str] | None,
        "Confidential Content"
    ],
    typical_examples: Annotated[List[str] | None, "Confidential Content"],
):
    """Confidential Content"""
    pass


@tool
@general_mutate_strategy(DeleteStrategy)
def delete_narrative_strategy(
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    strategy_to_delete: Annotated[
        str,
        "Confidential Content",
    ],
):
    """Confidential Content"""
    pass


@tool
@general_mutate_strategy(DeleteStrategy)
def delete_hook_strategy(
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    strategy_to_delete: Annotated[str, "Confidential Content"],
):
    """Confidential Content"""
    pass


@tool
@general_mutate_strategy(UpdateNarrativeStrategy)
def update_narrative_strategy(
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    current_strategy_name: Annotated[
        str,
        "Confidential Content",
    ],
    new_strategy_name: Annotated[
        str,
        "Confidential Content",
    ],
    updated_strategy_description: Annotated[
        Optional[str],
        "Confidential Content",
    ],
    updated_structure_flow_description: Annotated[
        Optional[str],
        "Confidential Content",
    ],
):
    """Confidential Content"""
    pass


@tool
@general_mutate_strategy(UpdateHookStrategy)
def update_hook_strategy(
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    current_strategy_name: Annotated[str, "The name of the hook strategy to be updated"],
    new_strategy_name: Annotated[
        str,
        "Confidential Content",
    ],
    *,
    updated_strategy_description: Annotated[
        Optional[str],
        "Confidential Content",
    ],
    updated_hook_selection_formula: Annotated[
        Optional[str],
        "Confidential Content",
    ],
    updated_audience_emotion: Annotated[
        Optional[List[str]],
        "Confidential Content",
    ],
    updated_pacing_structure: Annotated[
        Optional[List[str]],
        "Confidential Content",
    ],
    updated_visual_elements: Annotated[
        Optional[List[str]],
        "Confidential Content",
    ],
    updated_typical_examples: Annotated[
        Optional[List[str]],
        "Confidential Content",
    ],
):
    """Confidential Content"""
    pass

@tool
@general_mutate_strategy(SplitNarrativeStrategy)
def split_narrative_strategy(
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    strategy_name: Annotated[
        str,
        "Confidential Content",
    ],
    sub_strategies: Annotated[
        list[NarrativeStrategy],
        "Confidential Content",
    ],
):
    """Confidential Content"""

    pass


@tool
@general_mutate_strategy(SplitHookStrategy)
def split_hook_strategy(
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    strategy_name: Annotated[str, "Confidential Content"],
    sub_strategies: Annotated[list[HookStrategy], "Confidential Content"],
):
    """Confidential Content"""
    pass


@tool
@general_mutate_strategy(MergeNarrativeStrategy)
def merge_narrative_strategy(
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    strategy_names_to_merge: Annotated[
        list[str],
        "Confidential Content",
    ],
    merged_strategy_name: Annotated[
        str,
        "Confidential Content",
    ],
    strategy_description: Annotated[
        str,
        "Confidential Content",
    ],
    merged_structure_flow_description: Annotated[
        str,
        "Confidential Content",
    ],
):
    """Confidential Content"""
    pass


@tool
@general_mutate_strategy(MergeHookStrategy)
def merge_hook_strategy(
    state: Annotated[State, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    strategy_names_to_merge: Annotated[
        list[str], "Confidential Content"
    ],
    merged_strategy_name: Annotated[str, "Confidential Content"],
    strategy_description: Annotated[str, "Confidential Content"],
    merged_hook_selection_formula: Annotated[
        str, "Confidential Content"
    ],
    merged_audience_emotion: Annotated[List[str], "Confidential Content"],
    merged_pacing_structure: Annotated[List[str], "Confidential Content"],
    merged_visual_elements: Annotated[
        List[str], "Confidential Content"
    ],
    merged_typical_examples: Annotated[
        List[str], "Confidential Content"
    ],
):
    """Confidential Content"""
    pass


STRATEGY_CONFIG_MAP = {
    "narrative": {
        "strategy_type": NarrativeStrategy,
        "prompt": PromptTemplate.from_template(template=NARRATIVE_SYSTEM_PROMPT, template_format="jinja2"),
        "tools": [
            add_narrative_strategy,
            delete_narrative_strategy,
            update_narrative_strategy,
            split_narrative_strategy,
            merge_narrative_strategy,
        ],
        "video_item_type": VideoItemWithDeliveryStyle,
        "data_filter": lambda df: df[df["valid_storyline"] == True],
    },
    "hook": {
        "strategy_type": HookStrategy,
        "prompt": PromptTemplate.from_template(template=HOOK_SYSTEM_PROMPT, template_format="jinja2"),
        "tools": [
            add_hook_strategy,
            delete_hook_strategy,
            create_strategy_tool(UpdateHookStrategy),
            split_hook_strategy,
            merge_hook_strategy,
        ],
        "video_item_type": VideoItemWithHook,
        "data_filter": lambda df: df[df["valid_hook"] == True].drop_duplicates(subset=["project_id"]),
    },
}
