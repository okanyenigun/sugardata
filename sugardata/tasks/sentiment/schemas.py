from pydantic import BaseModel, Field


class OntologyConcept(BaseModel):
    generated_concept: str = Field(title="Generated Concept", description="The generated sub or related concepts")
    explanation: str = Field(title="Explanation", description="A brief explanation of the generated concept")


class OntologyConcepts(BaseModel):
    concepts: list[OntologyConcept] = Field(title="Concepts", description="The generated sub or related concepts")


class WritingStyle(BaseModel):
    medium: str = Field(title="Medium", description="The medium where the sentence will be taken, the title")
    persona: str = Field(title="Persona", description="A persona (author perspective) that would naturally use this style")
    writing_style: str = Field(title="Writing Style", description="A writing style description, which may include tone, formality, and influences (e.g., technical, poetic, first-person, etc.)")


class WritingStyles(BaseModel):
    styles: list[WritingStyle] = Field(title="Styles", description="The generated writing styles")


class GeneratedText(BaseModel):
    text: str = Field(title="Generated Text", description="The generated text based on the input parameters.")
    id: str = Field(title="ID", description="The unique identifier of the generated text.")