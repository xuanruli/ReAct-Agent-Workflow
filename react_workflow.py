import pickle
import random
from functools import partial
from pathlib import Path

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config
from langgraph.func import entrypoint, task
from langgraph.prebuilt import create_react_agent
from langgraph.runtime import Runtime
from langgraph.types import CachePolicy, RetryPolicy, interrupt
from langgraph.typing import ContextT
from react_tools import STRATEGY_CONFIG_MAP
from react_types import State
from react_types import BaseStrategy
from utils.window_utils import VideoItem, context_window_size, truncate_summary

llm_gpt_light = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.1,
    use_responses_api=True,
    output_version="responses/v1",
)

llm_gpt_pro = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0.1,
    reasoning_effort="medium",
    use_responses_api=True,
    output_version="responses/v1",
)

llm_gemini_pro = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.1,
)
llm_gemini_light = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
)


parent_path = Path(__file__).parent
df = pd.read_csv(parent_path / "filtered_tt&yt_longtrans+sp&hook_all_lang.csv").fillna("")


def dynamic_llm_callable(
    state: State,
    _runtime: Runtime[ContextT],
    llm_light: BaseChatModel,
    llm_pro: BaseChatModel,
    tools: list[StructuredTool],
    window_size: int = 90000,
) -> BaseChatModel:
    # [0] is system message as context window buffer
    context_size = context_window_size(state.messages[0].content)
    if context_size < window_size:
        print(f"---Context size: {context_size} tokens < window size: {window_size} tokens, using light model")
        return llm_light.bind_tools(tools)
    else:
        print(f"---Context size: {context_size} tokens >= window size: {window_size} tokens, using reasoning model")
        return llm_pro.bind_tools(tools)


# lazy render prompt for dynamic strategy set change during react loop
def dynamic_prompt_callable(
    state: State,
    sys_prompt_template: PromptTemplate,
    video_clips: list[VideoItem],
    max_strategies: int,
) -> str:

    existing_strategy_set = state.strategy_set
    return sys_prompt_template.format(
        len=len,
        strategy_set=existing_strategy_set,
        video_clips=video_clips,
        max_strategies=max_strategies,
    )


@task
async def process_video_clips_batch(
    existing_strategy_set: list[BaseStrategy],
    strategy_type: str,
    video_clips: list[VideoItem],
    batch_idx: int,
    max_strategies: int,
    previous_batches_summary: str = "",
) -> tuple[list[BaseStrategy], str]:

    sys_prompt_template = STRATEGY_CONFIG_MAP[strategy_type]["prompt"]

    tools = STRATEGY_CONFIG_MAP[strategy_type]["tools"]

    model_callable = partial(
        dynamic_llm_callable,
        llm_light=llm_gemini_light,
        llm_pro=llm_gemini_pro,
        tools=tools,
    )

    prompt_callable = partial(
        dynamic_prompt_callable,
        sys_prompt_template=sys_prompt_template,
        video_clips=video_clips,
        max_strategies=max_strategies,
    )

    # agent_parser = create_react_agent_parser(llm)
    agent = create_react_agent(
        model=model_callable,
        prompt=prompt_callable,
        tools=[*tools],
        state_schema=State,
        checkpointer=InMemorySaver(),
    )

    # smart context window injection
    window_messages = [HumanMessage(content="Start batch video analysis.")] if not previous_batches_summary else []
    if previous_batches_summary:

        window_messages.append(
            SystemMessage(
                content=(
                    "This is a continuation of previous work."
                    "Here is a summary of what was accomplished in the previous batchs:\n---\n"
                    f"{previous_batches_summary}\n---\n"
                    "Use this context window for better understanding of previous batch."
                )
            )
        )
    # Run agent without streaming; execute once per batch
    await agent.ainvoke(
        {
            "strategy_set": existing_strategy_set,
            "messages": window_messages,
        },
        config=get_config(),
    )

    # smart context window acquisition
    output_state = agent.get_state(config=get_config())
    final_strategies = output_state.values["strategy_set"]
    output_messages = output_state.values["messages"]

    current_batch_summary = ""
    if output_messages and isinstance(output_messages[-1], AIMessage):
        content = output_messages[-1].content
        if isinstance(content, list):
            current_batch_summary = "\n".join(map(str, content))
        else:
            current_batch_summary = str(content)

    # save to pickle
    current_parent_path = Path(__file__).parent
    output_path = current_parent_path / f"strategy_set_{batch_idx}.pkl"
    with output_path.open("wb") as f:
        pickle.dump(output_state.values["strategy_set"], f)

    return final_strategies, current_batch_summary


@task
async def get_random_clip_batch() -> list[VideoItem]:
    config = get_config()

    batch_size = config["metadata"]["batch_size"]
    strategy_type = config["metadata"]["strategy_type"]
    videoitem = STRATEGY_CONFIG_MAP[strategy_type]["video_item_type"]
    data_filter = STRATEGY_CONFIG_MAP[strategy_type]["data_filter"]

    items = videoitem.get_video_items(df.pipe(data_filter))

    return random.sample(items, batch_size)


@entrypoint(checkpointer=InMemorySaver())
async def clip_analysis_workflow(
    input_tuple: tuple[list[BaseStrategy], str]
):

    config = get_config()
    max_iter: int = config["metadata"]["max_iter"]
    should_stop: bool = config["metadata"]["should_stop"]
    max_strategies: int = config["metadata"].get("max_strategies", 33)
    strategy_type: str = config["metadata"]["strategy_type"]
    video_item_type = STRATEGY_CONFIG_MAP[strategy_type]["video_item_type"]
    strategy_set, smart_context_window = input_tuple

    if strategy_set:
        assert all(
            isinstance(strategy, STRATEGY_CONFIG_MAP[strategy_type]["strategy_type"]) for strategy in strategy_set
        ), "Strategy set must be a list of strategies of the same type"

    print(f"Current max # of iterations: {max_iter}")

    num_iter = 0
    while not should_stop or num_iter < max_iter:
        if num_iter >= max_iter:
            interrupt(f"Break at iteration {num_iter}")

        clip_batch: list[VideoItem] = await get_random_clip_batch()

        assert all(
            isinstance(video_clip, video_item_type) for video_clip in clip_batch
        ), f"expected video type: {video_item_type}, but got: {type(clip_batch[0])} for strategy type: {strategy_type}"

        # process the video clips in batches
        print(f"== Processing batch {num_iter + 1} of size {len(clip_batch)} ==")
        strategy_set, batch_summary = await process_video_clips_batch(
            strategy_set, strategy_type, clip_batch, num_iter, max_strategies, smart_context_window
        )
        print(f"== Strategy set length after batch {num_iter + 1}: {len(strategy_set)} ==")
        smart_context_window += batch_summary
        smart_context_window = truncate_summary(smart_context_window, max_tokens=100000)

        current_parent_path = Path(__file__).parent
        output_path = current_parent_path / f"context_window.pkl"
        with output_path.open("wb") as f:
            pickle.dump(smart_context_window, f)

        print(f"== Smart context window length after batch {num_iter + 1}: {len(smart_context_window)} ==")
        num_iter += 1

    return strategy_set
