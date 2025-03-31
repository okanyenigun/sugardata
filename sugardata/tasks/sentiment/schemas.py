from typing import List
from pydantic import BaseModel, Field


class OntologyConcept(BaseModel):
    generated_concept: str = Field(title="Generated Concept", description="The generated sub or related concepts")
    explanation: str = Field(title="Explanation", description="A brief explanation of the generated concept")


class OntologyConcepts(BaseModel):
    concepts: list[OntologyConcept] = Field(title="Concepts", description="The generated sub or related concepts")
    

class GeneratedText(BaseModel):
    text: str = Field(title="Generated Text", description="The generated text based on the input parameters.")
    id: str = Field(title="ID", description="The unique identifier of the generated text.")


class Aspect(BaseModel):
    aspects: List[str] = Field(description="List of aspects related to the concept.")