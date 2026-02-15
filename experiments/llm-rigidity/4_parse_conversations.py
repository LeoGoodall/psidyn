import json
from pathlib import Path

BATCH_RESULTS_JSONL = Path("llm-rigidity/batch_results.jsonl")
OUTPUT_JSON = Path("llm-rigidity/conversations.json")

def parse_batch_results():
    """Parse batch results and extract all conversations."""
    conversations = []
    
    if not BATCH_RESULTS_JSONL.exists():
        print(f"Error: {BATCH_RESULTS_JSONL} not found")
        return
    
    with open(BATCH_RESULTS_JSONL, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                result = json.loads(line)
                
                # Check if response was successful
                if result.get("response", {}).get("status_code") != 200:
                    print(f"Warning: Skipping line {line_num} - non-200 status code")
                    continue
                
                # Navigate to the conversation data
                response_body = result.get("response", {}).get("body", {})
                output = response_body.get("output", [])
                
                # Find the message output (usually the second item after reasoning)
                message_output = None
                for item in output:
                    if item.get("type") == "message":
                        message_output = item
                        break
                
                if not message_output:
                    print(f"Warning: Skipping line {line_num} - no message output found")
                    continue
                
                # Extract the text content
                content = message_output.get("content", [])
                text_content = None
                for content_item in content:
                    if content_item.get("type") == "output_text":
                        text_content = content_item.get("text")
                        break
                
                if not text_content:
                    print(f"Warning: Skipping line {line_num} - no text content found")
                    continue
                
                # Parse the JSON string to get the conversation
                conversation_data = json.loads(text_content)
                
                # Add metadata from the batch result
                conversation_data["custom_id"] = result.get("custom_id")
                conversation_data["batch_request_id"] = result.get("id")
                
                conversations.append(conversation_data)
                
            except json.JSONDecodeError as e:
                print(f"Error: Failed to parse JSON on line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error: Unexpected error on line {line_num}: {e}")
                continue
    
    # Write all conversations to output file
    with open(OUTPUT_JSON, "w") as f:
        json.dump(conversations, f, indent=2)
    
    print(f"Successfully parsed {len(conversations)} conversations")
    print(f"Output saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    parse_batch_results()
