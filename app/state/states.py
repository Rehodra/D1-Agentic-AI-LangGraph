#now here we'll create a state machine for our application. This will help us manage the different states of our app and transition between them smoothly. We'll define a base state class and then create specific states that inherit from it. Each state will have its own behavior and can handle events accordingly.

import os

#1 type DICT:
from typing import TypedDict

class State(TypedDict):
    topic: str
    summary: str
    score: str

#2 pydantic use
from pydantic import BaseModel, field_validator

class State_Pydentic(BaseModel):
    topic: str
    summary: str
    score: str

    @field_validator('score')
    def validate_score(cls, v):
        if not v.isdigit() or not (0 <= int(v) <= 100):
            raise ValueError('Score must be a number between 0 and 100')
        return v

#3 standard python dataclass but it is used very rarelty 

from dataclasses import dataclass, field 

@dataclass
class State:
    topic : str = ""
    summary  : str = ""
    messages : list = field(default_factory=list)

#4 LangGraph message state:
from langgraph.graph import MessagesState

class State(MessagesState):
    # messages field is already included with add_messages reducer
    # just add your extra fields
    user_name: str
    language: str
