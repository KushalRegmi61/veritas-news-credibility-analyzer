# Pydantic schema for user input
from pydantic import BaseModel, computed_field, field_validator, Field
from typing import Annotated

class UserInput(BaseModel):
    text: Annotated[str, Field(..., description="User input text to analyze.",
                               examples=["Breaking News: AI outperforms humans!"])]


    @field_validator("text")
    def check_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("Text input must not be empty or whitespace.")
        return v
    
    @computed_field
    @property
    def text_length(self) -> int: 
        return len(self.text)   
    
    @computed_field
    @property
    def exclamations_mark_count(self) -> int:
        return str(self.text).count('!')
    
    @computed_field
    @property
    def questions_mark_count(self) -> int: 
        return str(self.text).count('?' or r'\?')

    
    @computed_field
    @property
    def uppercase_words_count(self) -> int: 
        return sum(1 for w in str(self.text).split() if w.isupper())


