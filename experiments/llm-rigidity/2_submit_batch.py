import os
from pathlib import Path
import openai

openai.api_key = os.getenv("OPENAI_API_KEY_PSIDYN")

BATCH_FILE = Path("llm-rigidity/conversation_batch.jsonl")
BATCH_ID_FILE = Path("llm-rigidity/latest_batch_id.txt")

def main():

    f = openai.files.create(
        file=open(BATCH_FILE, "rb"),
        purpose="batch"
    )
    file_id = f.id

    batch = openai.batches.create(
        input_file_id=file_id,
        endpoint="/v1/responses",
        completion_window="24h"
    )

    BATCH_ID_FILE.write_text(batch.id)

if __name__ == "__main__":
    main()
