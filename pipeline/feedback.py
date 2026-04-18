import json
import os
import uuid

def save_feedback(llm_output, feedback):
    record = {
        "id": str(uuid.uuid4()),
        "llm_output": llm_output,
        "feedback": feedback
    }

    file_path = "data/feedback_store.json"

    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            # 🔥 fix if it's a dict instead of list
            if isinstance(data, dict):
                data = [data]

        except:
            data = []
    else:
        data = []

    data.append(record)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)