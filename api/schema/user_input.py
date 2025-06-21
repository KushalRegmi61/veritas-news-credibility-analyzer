# Pydantic schema for user input
from pydantic import BaseModel

class UserInput(BaseModel):
    text: str
