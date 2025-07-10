# Pydantic schema for user input
from pydantic import BaseModel, computed_field, field_validator, Field
from typing import Annotated

class UserInput(BaseModel):
    text: Annotated[str, Field(
        ...,
        description="Paste the full news text here -----> \nTips: Avoid using unescaped double quotes.",
        json_schema_extra = {
            "example": (
                
                    "On July 4, 2025, the United States experienced a complex Independence Day marked by both celebrations and unrest. "
                    "Millions participated in the \"Free America Weekend\" protests, opposing the recently signed \"One Big Beautiful Bill.\""
                
            )
        }
    )]

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


