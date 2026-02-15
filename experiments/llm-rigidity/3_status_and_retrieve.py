import os
from pathlib import Path
import openai

openai.api_key = os.getenv("OPENAI_API_KEY_PSIDYN")

BATCH_ID_FILE = Path("llm-rigidity/latest_batch_id.txt")
OUTPUT_JSONL = Path("llm-rigidity/batch_results.jsonl")
ERROR_JSONL = Path("llm-rigidity/batch_errors.jsonl")

def main():

    batch_id = BATCH_ID_FILE.read_text().strip()
    info = openai.batches.retrieve(batch_id)

    print(info.status)

    if info.status == "completed":
        if info.output_file_id:
            content = openai.files.content(info.output_file_id)
            text = content.text
            OUTPUT_JSONL.write_text(text)
            print(f"Output file saved to {OUTPUT_JSONL}")
        
        if info.error_file_id:
            content = openai.files.content(info.error_file_id)
            text = content.text
            ERROR_JSONL.write_text(text)
            print(f"Error file saved to {ERROR_JSONL}")

if __name__ == "__main__":
    main()