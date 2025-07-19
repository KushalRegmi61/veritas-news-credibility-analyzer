from pydantic import BaseModel, computed_field, field_validator, Field
from typing import Annotated
import textstat
import re



class UserInput(BaseModel):
    text: Annotated[str, Field(
        ...,
        description="Paste the full news text here.",
        json_schema_extra={
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
    def uppercase_words_count(self) -> int:
        return sum(word.isupper() for word in self.text.split())

    @computed_field
    @property
    def has_uppercase_emphasis(self) -> int:
        return int(self.uppercase_words_count > 0)

    @computed_field
    @property
    def readability_score(self) -> float:
        try:
            return textstat.flesch_reading_ease(self.text)
        except Exception:
            return 0.0

    @computed_field
    @property
    def long_text_flag(self) -> int:
        return int(self.text_length > 3000)

    @computed_field
    @property
    def text_length_bin(self) -> int:
        if self.text_length < 1000:
            return 0
        elif self.text_length < 2000:
            return 1
        elif self.text_length < 3000:
            return 2
        else:
            return 3

    @computed_field
    @property
    def punctuation_alert(self) -> int:
        return int(bool(re.search(r'[!?]', self.text)))



    @computed_field
    @property
    def first_sentence_length(self) -> int:
        first_sentence = re.split(r'[.!?]', self.text.strip())[0]
        return len(first_sentence.split())


# Optional CLI test
if __name__ == "__main__":
    sample_input = UserInput(
        **{
            "text": (
                "BREAKING: The United States experienced a complex Independence Day. "
                "Millions joined 'Free America Weekend' protests, citing opposition to the newly signed 'One Big Beautiful Bill.'"
            )
        }
    )
    print(sample_input.model_dump(mode="json"))
