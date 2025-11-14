import copy
from typing import List, Dict, Any


def build_base_messages(system_prompt: str = "", user_prompt: str = "") -> List[Dict[str, Any]]:
    """
    Constructs the base conversation structure shared by every attack step.
    The user turn always contains an image placeholder so downstream chat
    templates will insert the proper visual tokens.
    """
    messages: List[Dict[str, Any]] = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        })

    user_content = [{"type": "image"}]
    if user_prompt:
        user_content.append({"type": "text", "text": user_prompt})

    messages.append({
        "role": "user",
        "content": user_content
    })

    return messages


def append_assistant_response(messages: List[Dict[str, Any]], response: str) -> List[Dict[str, Any]]:
    """
    Returns a deep copy of messages with the provided assistant response appended.
    """
    convo = copy.deepcopy(messages)
    convo.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response}]
    })
    return convo
