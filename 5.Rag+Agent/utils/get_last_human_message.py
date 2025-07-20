from langchain.schema import BaseMessage, HumanMessage
from typing import List


def get_last_human_message(messages: List[BaseMessage]) -> HumanMessage:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message
    print('error')
    exit(0)