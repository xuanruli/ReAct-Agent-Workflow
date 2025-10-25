from typing import List, Type, TypeVar

import pandas as pd
import tiktoken
from pydantic import BaseModel
from react_types import BaseStrategy

T = TypeVar("T", bound="VideoItem")


class VideoItem(BaseModel):
    video_id: str
    genre: str
    transcript: str

    @staticmethod
    def process_transcript_with_visual(text: str) -> str:
        tr_chunks = text.split("__visual")
        processed: list[tuple[str, str]] = []

        if len(tr_chunks) == 1:
            processed.append(("text", tr_chunks[0]))
        else:
            for chunk in tr_chunks:
                if len(chunk) == 0:
                    continue

                try:
                    dot_index = chunk.index(".")
                    visual_tr = chunk[:dot_index].strip()
                    text_tr = chunk[dot_index + 1 :].strip() if dot_index < len(chunk) - 1 else ""
                    if len(visual_tr) > 0:
                        processed.append(("visual", visual_tr))
                    if len(text_tr) > 0:
                        processed.append(("text", text_tr))
                except ValueError:
                    processed.append(("visual", chunk))

        return "\n".join([f"[{tr_type}] {tr_content}" for tr_type, tr_content in processed])

    @property
    def processed_transcripts(self) -> str:
        return self.process_transcript_with_visual(self.transcript)

    @classmethod
    def get_video_items(cls, df: pd.DataFrame) -> list[T]:
        return [cls.from_row(row) for _, row in df.iterrows()]


class VideoItemWithHook(VideoItem):
    hook: str

    @property
    def processed_hook(self) -> str:
        return self.process_transcript_with_visual(self.hook)

    @classmethod
    def from_row(cls, row: pd.Series) -> "VideoItemWithHook":
        return cls(
            video_id=row["video_id"],
            genre=row["genre"],
            transcript=row["long_transcript"],
            hook=row["hook"],
        )


class VideoItemWithDeliveryStyle(VideoItem):
    delivery_style: str
    subgenre: str

    @classmethod
    def from_row(cls, row: pd.Series) -> "VideoItemWithDeliveryStyle":
        return cls(
            video_id=row["video_id"],
            genre=row["genre"],
            subgenre=row["subgenre"] if row["subgenre"] else "",
            transcript=row["full_transcript"],
            delivery_style="",
        )


def split_train_test(df, test_size=0.2):
    train_df = df.sample(frac=1 - test_size, random_state=42)
    test_df = df.drop(train_df.index)
    return train_df, test_df


def parse_strategy_set_into_json(strategy_set: List[BaseStrategy]):
    return [s.model_dump() for s in strategy_set]


def parse_json_into_strategy_set(json_data: List[dict], pydantic_model: Type[BaseStrategy]) -> List[BaseStrategy]:
    return [pydantic_model.model_validate(data) for data in json_data]


def id_to_youtube_url(id):
    return f"https://www.youtube.com/watch?v={id}"


def truncate_summary(summary: str, max_tokens: int = 1000, model_name="gpt-4o"):
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(summary)
    if len(tokens) > max_tokens:
        truncated = enc.decode(tokens[:max_tokens])
        return truncated + "\n...[truncated]..."
    return summary


def context_window_size(context_window: str, max_tokens: int = 100000, model_name="gpt-4o") -> int:
    enc = tiktoken.encoding_for_model(model_name)
    tokens = enc.encode(context_window)
    return len(tokens)
