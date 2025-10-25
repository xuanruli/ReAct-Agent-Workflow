from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, List, Optional, TypeVar
from langgraph.prebuilt import InjectedState
from typing_extensions import Annotated
from pydantic import BaseModel, Field
from langchain_core.tools import InjectedToolCallId
from langgraph.prebuilt.chat_agent_executor import AgentStatePydantic
from typing_extensions import Annotated

class BaseStrategy(BaseModel):
    strategy_name: str = Field(..., description="The name of the clipping strategy")
    strategy_description: str = Field(..., description="The description and rational of the clipping strategy")

    @property
    def id(self) -> str:
        return self.strategy_name


class RemoveStrategy(BaseModel):
    """Confidential Content"""

    strategy_name: str = Field(..., description="The name of the strategy to remove")

    @property
    def id(self) -> str:
        return self.strategy_name


StrategyItem = BaseStrategy | RemoveStrategy



def merge_strategy_set(
    old_strategy_set: list[StrategyItem], new_strategy_set: list[StrategyItem]
) -> list[StrategyItem]:
    """
    Confidential Content
    """
    # Start with a copy of old strategies
    merged = old_strategy_set.copy()
    assert all(isinstance(strategy, BaseStrategy) for strategy in merged), "Should not happen"

    merged_by_id = {strategy.id: i for i, strategy in enumerate(merged)}
    ids_to_remove = set()

    # Process each item in the new set
    for item in new_strategy_set:
        if isinstance(item, RemoveStrategy):
            # Mark for removal
            if item.id in merged_by_id:
                ids_to_remove.add(item.id)
            else:
                raise ValueError(f"Attempting to delete a strategy that doesn't exist: {item.strategy_name}")
        else:
            # It's a regular strategy
            if (existing_idx := merged_by_id.get(item.id)) is not None:
                # Strategy exists - update it
                ids_to_remove.discard(item.id)
                merged[existing_idx] = item
            else:
                # Strategy doesn't exist - add it
                merged_by_id[item.id] = len(merged)
                merged.append(item)

    # Remove strategies marked for deletion
    return [strategy for strategy in merged if strategy.id not in ids_to_remove]


class State(AgentStatePydantic):
    """Confidential Content"""

    strategy_set: Annotated[list[StrategyItem], merge_strategy_set] = Field(
        ...,
        description="The current strategy set",
    )

class StrategyTool(ABC):
    @abstractmethod
    def mutate(self, strategy_set: list[StrategyItem]) -> list[StrategyItem]:
        raise NotImplementedError


class NarrativeStrategy(BaseStrategy):
    structure_flow_description: str = Field(
        default="", description="Confidential Content"
    )


class HookStrategy(BaseStrategy):
    hook_selection_formula: str = Field(
        ..., description="Confidential Content"
    )
    audience_emotion: List[str] = Field(
        ...,
        description="Confidential Content"
    )
    pacing_structure: List[str] = Field(
        ...,
        description="Confidential Content"
    )
    visual_elements: List[str] = Field(
        default_factory=list,
        description="Confidential Content"
    )
    typical_examples: List[str] = Field(
        default_factory=list, max_length=4, description="Confidential Content"
    )


class AddNarrativeStrategy(BaseStrategy, StrategyTool):
    """Confidential Content"""

    structure_flow_description: str = Field(
        default="", description="Confidential Content"
    )

    def mutate(self, strategy_set: list[StrategyItem]) -> list[StrategyItem]:
        if self.strategy_name in [strategy.strategy_name for strategy in strategy_set]:
            raise ValueError(f"Strategy {self.strategy_name} already exists in strategy set")

        return [
            NarrativeStrategy(
                strategy_name=self.strategy_name,
                strategy_description=self.strategy_description,
                structure_flow_description=self.structure_flow_description,
            )
        ]


class AddHookStrategy(BaseStrategy, StrategyTool):
    """Confidential Content"""

    hook_selection_formula: str = Field(
        ...,
        description="Confidential Content"
    )
    audience_emotion: List[str] = Field(
        ...,
        description="Confidential Content"
    )
    pacing_structure: List[str] = Field(
        ...,
        description="Confidential Content"
    )
    visual_elements: List[str] = Field(
        default_factory=list,
        description="Confidential Content"
    )
    typical_examples: List[str] = Field(default_factory=list, description="The typical examples of the hook strategy")

    def mutate(self, strategy_set: list[StrategyItem]) -> list[StrategyItem]:
        if self.strategy_name in [strategy.strategy_name for strategy in strategy_set]:
            raise ValueError(f"Strategy {self.strategy_name} already exists in strategy set")

        return [
            HookStrategy(
                strategy_name=self.strategy_name,
                strategy_description=self.strategy_description,
                hook_selection_formula=self.hook_selection_formula,
                audience_emotion=self.audience_emotion,
                pacing_structure=self.pacing_structure,
                visual_elements=self.visual_elements,
                typical_examples=self.typical_examples,
            )
        ]


class DeleteStrategy(BaseModel, StrategyTool):
    """Confidential Content"""

    strategy_to_delete: str = Field(..., description="The name of the narrative strategy to delete")

    def mutate(self, strategy_set: list[StrategyItem]) -> list[RemoveStrategy]:
        if self.strategy_to_delete not in [strategy.strategy_name for strategy in strategy_set]:
            raise ValueError(f"Strategy {self.strategy_to_delete} not found in strategy set")

        # Return the original strategy set plus a removal marker
        # The merge function will handle the actual removal
        return [RemoveStrategy(strategy_name=self.strategy_to_delete)]


T = TypeVar("T", bound=StrategyItem)


class UpdateStrategyBase(BaseModel, StrategyTool, Generic[T]):
    """
    Confidential Content
    """
    state: Annotated[State, InjectedState] = Field(..., description="The state of the clip strategy analysis agent")
    tool_call_id: Annotated[str, InjectedToolCallId] = Field(..., description="The tool call id")
    current_strategy_name: str = Field(..., description="The name of the strategy to be updated")
    new_strategy_name: str = Field(
        None,
        description="Confidential Content"
    )
    updated_strategy_description: Optional[str] = Field(
        None,
        description="Confidential Content"
    )

    @abstractmethod
    def create_updated_strategy(self) -> T:
        raise NotImplementedError

    def mutate(self, strategy_set: list[StrategyItem]) -> list[StrategyItem]:
        existing_names = {strategy.strategy_name for strategy in strategy_set}
        if self.current_strategy_name not in existing_names:
            raise ValueError(f"Strategy {self.current_strategy_name} not found in strategy set")

        current_strategy = next(
            strategy for strategy in strategy_set if strategy.strategy_name == self.current_strategy_name
        )
        is_renamed = self.new_strategy_name and self.new_strategy_name != self.current_strategy_name
        updated_strategy = self.create_updated_strategy(current_strategy, is_renamed)

        if is_renamed:
            return [
                RemoveStrategy(strategy_name=self.current_strategy_name),
                updated_strategy,
            ]
        return [
            updated_strategy,
        ]


class UpdateNarrativeStrategy(UpdateStrategyBase[NarrativeStrategy]):

    updated_structure_flow_description: Optional[str] = Field(
        None,
        description="Confidential Content"
    )

    def create_updated_strategy(self, current_strategy: NarrativeStrategy, is_renamed: bool) -> NarrativeStrategy:
        return NarrativeStrategy(
            strategy_name=self.new_strategy_name if is_renamed else current_strategy.strategy_name,
            strategy_description=(
                self.updated_strategy_description
                if self.updated_strategy_description
                else current_strategy.strategy_description
            ),
            structure_flow_description=self.updated_structure_flow_description
            or current_strategy.structure_flow_description,
        )


class UpdateHookStrategy(UpdateStrategyBase[HookStrategy]):
    """
    Confidential Content
    """
    updated_hook_selection_formula: Optional[str] = Field(
        None,
        description="Confidential Content"
    )
    updated_audience_emotion: Optional[List[str]] = Field(
        None,
        description="Confidential Content"
    )
    updated_pacing_structure: Optional[List[str]] = Field(
        None,
        description="Confidential Content"
    )
    updated_visual_elements: Optional[List[str]] = Field(
        None,
        description="Confidential Content"
    )
    updated_typical_examples: Optional[List[str]] = Field(
        None,
        max_length=4,
        description="Confidential Content"
    )

    def create_updated_strategy(self, current_strategy: HookStrategy, is_renamed: bool) -> HookStrategy:
        return HookStrategy(
            strategy_name=self.new_strategy_name if is_renamed else current_strategy.strategy_name,
            strategy_description=(
                self.updated_strategy_description
                if self.updated_strategy_description
                else current_strategy.strategy_description
            ),
            hook_selection_formula=(
                self.updated_hook_selection_formula
                if self.updated_hook_selection_formula
                else current_strategy.hook_selection_formula
            ),
            audience_emotion=(
                self.updated_audience_emotion if self.updated_audience_emotion else current_strategy.audience_emotion
            ),
            pacing_structure=(
                self.updated_pacing_structure if self.updated_pacing_structure else current_strategy.pacing_structure
            ),
            visual_elements=(
                self.updated_visual_elements if self.updated_visual_elements else current_strategy.visual_elements
            ),
            typical_examples=(
                self.updated_typical_examples if self.updated_typical_examples else current_strategy.typical_examples
            ),
        )


class SplitStrategyBase(BaseModel, StrategyTool, Generic[T]):
    strategy_name: str = Field(
        ...,
        description="Confidential Content"
    )

    sub_strategies: list[T] = Field(
        ...,
        description="Confidential Content"
    )

    @abstractmethod
    def create_sub_strategies(self) -> list[T]:
        raise NotImplementedError

    def mutate(self, strategy_set: list[StrategyItem]) -> list[StrategyItem]:
        if self.strategy_name not in [strategy.strategy_name for strategy in strategy_set]:
            raise ValueError(f"Strategy {self.strategy_name} not found in strategy set")

        existing_names = {strategy.strategy_name for strategy in strategy_set}
        sub_strategies = self.create_sub_strategies()
        sub_names = {s.strategy_name for s in sub_strategies}

        if existing_names & sub_names:
            conflict = ", ".join(sorted(existing_names & sub_names))
            raise ValueError(f"Sub-strategy name(s) already exist: {conflict}")

        # Return original strategy set + removal marker + new strategies
        return [RemoveStrategy(strategy_name=self.strategy_name), *sub_strategies]


class SplitNarrativeStrategy(SplitStrategyBase[NarrativeStrategy]):
    """
    Confidential Content
    """

    def create_sub_strategies(self) -> list[NarrativeStrategy]:
        return self.sub_strategies


class SplitHookStrategy(SplitStrategyBase[HookStrategy]):
    """
    Confidential Content
    """

    def create_sub_strategies(self) -> list[HookStrategy]:
        return self.sub_strategies


class MergeStrategyBase(BaseModel, StrategyTool, Generic[T]):
    strategy_names_to_merge: list[str] = Field(
        ...,
        description="Confidential Content"
    )

    merged_strategy_name: str = Field(
        ...,
        description="Confidential Content"
    )

    strategy_description: str = Field(
        ...,
        description="Confidential Content"
    )

    @abstractmethod
    def create_merged_strategy(self) -> T:
        raise NotImplementedError

    def mutate(self, strategy_set: list[StrategyItem]) -> list[StrategyItem]:
        for strategy_name in self.strategy_names_to_merge:
            if strategy_name not in [strategy.strategy_name for strategy in strategy_set]:
                raise ValueError(f"Strategy {strategy_name} not found in strategy set")

        existing_names = {strategy.strategy_name for strategy in strategy_set}
        if self.merged_strategy_name in existing_names:
            raise ValueError(f"Merged strategy name '{self.merged_strategy_name}' already exists")

        # Return original strategy set + removal markers + merged strategy
        removal_markers = [RemoveStrategy(strategy_name=name) for name in self.strategy_names_to_merge]
        merged_strategy = self.create_merged_strategy()

        return [*removal_markers, merged_strategy]


class MergeNarrativeStrategy(MergeStrategyBase[NarrativeStrategy]):
    """
    Confidential Content
    """

    merged_structure_flow_description: str = Field(
        ...,
        description="Confidential Content"
    )

    def create_merged_strategy(self) -> NarrativeStrategy:
        return NarrativeStrategy(
            strategy_name=self.merged_strategy_name,
            strategy_description=self.strategy_description,
            structure_flow_description=self.merged_structure_flow_description,
        )


class MergeHookStrategy(MergeStrategyBase[HookStrategy]):
    """
    Confidential Content
    """

    merged_hook_selection_formula: str = Field(
        ...,
        description="Confidential Content"
    )
    merged_audience_emotion: List[str] = Field(
        ...,
        description="Confidential Content"
    )
    merged_pacing_structure: List[str] = Field(
        ...,
        description="Confidential Content"
    )
    merged_visual_elements: List[str] = Field(
        ...,
        description="Confidential Content"
    )
    merged_typical_examples: List[str] = Field(
        ...,
        max_length=4,
        description="Confidential Content"
    )

    def create_merged_strategy(self) -> HookStrategy:
        return HookStrategy(
            strategy_name=self.merged_strategy_name,
            strategy_description=self.strategy_description,
            hook_selection_formula=self.merged_hook_selection_formula,
            audience_emotion=self.merged_audience_emotion,
            pacing_structure=self.merged_pacing_structure,
            visual_elements=self.merged_visual_elements,
            typical_examples=self.merged_typical_examples,
        )


AddStrategyBase = AddNarrativeStrategy | AddHookStrategy

ToolType = UpdateStrategyBase | SplitStrategyBase | DeleteStrategy | MergeStrategyBase | AddStrategyBase
