import os
import json
import uuid
from pathlib import Path

OUTPUT_JSONL = Path("llm-rigidity/conversation_batch.jsonl")
CONDITIONS = ["rigid-rigid", "rigid-flexible", "flexible-flexible"]
TOPICS = ["art", "politics", "food", "science", "morality"]
ITERATIONS = 100

def get_condition_description(condition: str, iteration: int):
    if condition == "rigid-flexible": # swap who starts each iteration
        if iteration % 2 == 0:
            return "Speaker A is cognitively rigid and Speaker B is flexible. Speaker A starts."
        else:
            return "Speaker A is cognitively rigid and Speaker B is flexible. Speaker B starts."
    if condition == "rigid-rigid":
        return "Both speakers behave with cognitive rigidity. Speaker A starts."
    if condition == "flexible-flexible":
        return "Both speakers behave with cognitive flexibility. Speaker A starts."

def build_prompt_body(prompt: str):
    return {
        "model": "gpt-5-nano",
        "input": prompt,
        "text": {
            "format": {
                "name": "conversation_schema",
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "conversation_id": {"type": "string"},
                        "condition": {"type": "string"},
                        "topic": {"type": "string"},
                        "turns": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "turn": {"type": "integer"},
                                    "speaker": {"type": "string"},
                                    "text": {"type": "string"}
                                },
                                "required": ["turn", "speaker", "text"],
                                "additionalProperties": False
                            }
                        }
                    },
                    "required": ["conversation_id", "condition", "topic", "turns"],
                    "additionalProperties": False
                }
            }
        }
    }

def build_prompt(condition: str, topic: str, conv_id: str, iteration: int):
    explanation = (
        "Cognitive rigidity is the difficulty in shifting from established patterns of thinking, feeling, or behaving when faced with new information or changing circumstances. "
        "Cognitive flexibility is the ability to adapt thinking and behavior in response to new information or environmental demands."
    )
    cond_text = get_condition_description(condition, iteration)

    return (
        f"Generate a 20-turn dialogue about {topic} between two agents who begin opposing each other's positions. Each turn should have 2–3 sentences.\n"
        f"Condition: {cond_text}\n"
        f"{explanation}\n"
        f"Return only valid JSON matching the schema.\n"
        f"conversation_id: {conv_id}\n"
        f"condition: {condition}\n"
        f"topic: {topic}"
    )

with open(OUTPUT_JSONL, "w") as out:
    for cond in CONDITIONS:
        for topic in TOPICS:
            for i in range(ITERATIONS):
                conv_id = str(uuid.uuid4())
                prompt = build_prompt(cond, topic, conv_id, i)
                entry = {
                    "custom_id": f"{cond}_{topic}_{i:03d}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": build_prompt_body(prompt)
                }
                out.write(json.dumps(entry) + "\n")
print(f"Batch file saved to {OUTPUT_JSONL}")