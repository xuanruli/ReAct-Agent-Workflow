import json
from datetime import datetime
from pathlib import Path
from typing import Type

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def create_jsonl_file_from_df(df: pd.DataFrame, columns: list[str], system_prompt: str, pydantic_model: BaseModel):
    batch_requests = []
    json_schema = pydantic_model.model_json_schema()
    json_schema["additionalProperties"] = False

    for idx, row in df.iterrows():

        user_text = []
        for col in columns:
            var = str(row[col])
            user_text.append(f"{col}: {var}")
        user_text = "\n".join(user_text)

        request = {
            "custom_id": f"request-{idx}",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": "gpt-4o-mini",
                "input": [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"\n{user_text}"}],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "strict": True,
                        "name": pydantic_model.__name__,
                        "schema": json_schema,
                    }
                },
                "temperature": 0,
            },
        }
        
        batch_requests.append(request)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_file_path = f"response_api_batch_{timestamp}.jsonl"

    with Path(batch_file_path).open("w", encoding="utf-8") as f:
        for request in batch_requests:
            f.write(json.dumps(request) + "\n")

    return batch_file_path


def submit_batch_api_job(batch_file_path, description="Response API Pattern Analysis"):
    client = get_client()
    try:
        with Path(batch_file_path).open("rb") as f:
            batch_input_file = client.files.create(file=f, purpose="batch")

        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/responses",
            completion_window="24h",
            metadata={"description": description},
        )

        return batch_job.id

    except Exception as e:
        print(f"ERROR in batch submit: {e}")
        return None


def process_with_response_api_batch(
    df: pd.DataFrame, columns: list[str], system_prompt: str, pydantic_model: BaseModel
):
    batch_file_path = create_jsonl_file_from_df(df, columns, system_prompt, pydantic_model)
    batch_id = submit_batch_api_job(batch_file_path)

    if batch_id:
        print("BATCH API submitted successfully!")
        print(f"task id: {batch_id}")
        print(f"check status: check_batch_status('{batch_id}')")
        print(f"download results: download_batch_results('{batch_id}', df)")

        with Path("response_api_batch_id.txt").open("a") as f:
            f.write(batch_id + "\n")
        print("batch_id appended to response_api_batch_id.txt")

        return batch_id
    return None


def check_batch_status(batch_id: str):
    client = get_client()
    batch_job = client.batches.retrieve(batch_id)
    if batch_job.error_file_id:
        print(f"error file id: {batch_job.error_file_id}")
        try:
            error_content = client.files.content(batch_job.error_file_id)
            print("error details:")
            print(error_content.text[:1000])
        except Exception as e:
            print(f"cannot read error file: {e}")
    if batch_job.status != "completed":
        print(f"task not completed, current status: {batch_job.status}")
        print(f"error info: {batch_job.errors}")

        return None
    return batch_job.status


def cancel_batch(batch_id: str):
    """
    Cancel a batch job

    Args:
        batch_id (str): The batch job ID to cancel

    Returns:
        bool: True if successfully cancelled, False otherwise
    """
    client = get_client()
    try:
        # First check the current status
        batch_job = client.batches.retrieve(batch_id)
        current_status = batch_job.status

        print(f"Current batch status: {current_status}")

        if current_status in ["validating", "in_progress", "finalizing"]:
            cancelled_batch = client.batches.cancel(batch_id)
            print(f"Batch {batch_id} has been cancelled successfully!")
            print(f"New status: {cancelled_batch.status}")
            return True
        if current_status == "cancelled":
            print(f"Batch {batch_id} is already cancelled.")
            return True
        if current_status == "completed":
            print(f"Batch {batch_id} is already completed and cannot be cancelled.")
            return False
        if current_status == "failed":
            print(f"Batch {batch_id} has already failed.")
            return False
        if current_status != "validating" and current_status != "in_progress" and current_status != "finalizing":
            print(f"Batch {batch_id} is in status '{current_status}' and cannot be cancelled.")
            return False

        return True

    except Exception as e:
        print(f"Error cancelling batch {batch_id}: {e}")
        return False


def download_batch_results(batch_id: str, original_df: pd.DataFrame, pydantic_model: Type[BaseModel]):
    client = get_client()
    try:
        batch_job = client.batches.retrieve(batch_id)

        if not check_batch_status(batch_id):
            return None

        error_dict = {}
        for keys in pydantic_model.model_fields:
            error_dict[keys] = "ERROR"

        result_file_id = batch_job.output_file_id
        file_response = client.files.content(result_file_id)

        results = {}
        for line in file_response.text.strip().split("\n"):
            if line:
                result = json.loads(line)
                custom_id = result["custom_id"]

                if result["response"]["status_code"] == 200:
                    try:
                        content = result["response"]["body"]["output"][0]["content"][0]["text"]
                        results[custom_id] = json.loads(content)
                        # content.as_model() can do pydantic validate and parse_raw() for response
                    except Exception as e:
                        results[custom_id] = error_dict
                        print(f"error: {e}")
                else:
                    results[custom_id] = error_dict

        pydantic_results = []
        for idx in range(len(original_df)):
            custom_id = f"request-{idx}"
            pydantic_results.append(results.get(custom_id, error_dict))

        result_df = pd.DataFrame(pydantic_results)
        final_df = pd.concat([original_df.reset_index(drop=True), result_df], axis=1)

        # use the first field to judge the success rate
        first_field = next(iter(pydantic_model.model_fields.keys())) if pydantic_model.model_fields else None
        success_count = len([r for r in pydantic_results if r.get(first_field) != "ERROR"]) if first_field else 0
        print(f"processing completed! success: {success_count}/{len(pydantic_results)}")

        return final_df

    except Exception as e:
        print(f"error: {e}")
        return None
