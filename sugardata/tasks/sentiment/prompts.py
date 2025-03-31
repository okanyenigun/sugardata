

CONCEPT_PREFIX = """
You are an expert in ontology design, knowledge representation, and concept hierarchy analysis. Your task is to analyze a given concept by identifying at least 10 related sub-concepts and adjacent concepts—some being hierarchical subcategories while others are creative connections beyond direct taxonomy.

For the given concept, generate a structured list of related concepts in the following format:

Sub-concepts (Hierarchical Breakdown): Concepts that fall directly under in a structured manner.
Related Concepts (Creative Connections): Concepts that are relevant but do not directly fall under the main concept. These may include adjacent fields, applications, or interdisciplinary connections.
Think through each aspect carefully, considering logical relationships, broader implications, and creative associations before finalizing the response.
"""

CONCEPT_SUFFIX = """
Identify at least 10 concepts related to concept, ensuring a mix of hierarchical sub-concepts and adjacent concepts.
Clearly distinguish between sub-concepts (which fall under concept) and related concepts (which connect creatively).
Format the response in a structured, bullet-point list with a short explanation of why each concept is relevant.

Output Format:

Concept: <Insert Given Concept>
- **Sub-concepts:**
  1. <Sub-concept 1> – <Brief Explanation>
  2. <Sub-concept 2> – <Brief Explanation>
  ...
  
- **Related Concepts:**
  6. <Related Concept 1> – <Brief Explanation>
  7. <Related Concept 2> – <Brief Explanation>
  ...

Format Instructions:
{format_instructions}

The given concept is {concept}
"""

CONCEPT_EXAMPLE_TEMPLATE = """
Concept: {concept}\n- {generated_concept}: {explanation}
"""

CONCEPT_EXAMPLES = [
    # Examples for Artificial Intelligence
    {
        "concept": "Artificial Intelligence",
        "generated_concept": "Machine Learning",
        "explanation": "A subset of AI that focuses on algorithms learning from data."
    },
    {
        "concept": "Artificial Intelligence",
        "generated_concept": "Deep Learning",
        "explanation": "A branch of machine learning using neural networks for complex tasks."
    },
    {
        "concept": "Artificial Intelligence",
        "generated_concept": "Neural Networks",
        "explanation": "Computational models inspired by the human brain for AI tasks."
    },
    {
        "concept": "Artificial Intelligence",
        "generated_concept": "Reinforcement Learning",
        "explanation": "An AI training method using rewards and penalties."
    },
    {
        "concept": "Artificial Intelligence",
        "generated_concept": "Symbolic AI",
        "explanation": "AI that relies on rules and logic instead of statistical models."
    },
    {
        "concept": "Artificial Intelligence",
        "generated_concept": "Computational Neuroscience",
        "explanation": "The study of brain computations, related to AI mechanisms."
    },
    {
        "concept": "Artificial Intelligence",
        "generated_concept": "Cognitive Science",
        "explanation": "Interdisciplinary field studying human intelligence and AI parallels."
    },
    {
        "concept": "Artificial Intelligence",
        "generated_concept": "Robotics",
        "explanation": "AI applications in autonomous machines and robots."
    },
    {
        "concept": "Artificial Intelligence",
        "generated_concept": "Ethics in AI",
        "explanation": "The study of AI’s moral and societal impacts."
    },
    {
        "concept": "Artificial Intelligence",
        "generated_concept": "Generative AI",
        "explanation": "AI that can create content, such as images or text, using deep learning."
    },

    # Examples for Climate Change
    {
        "concept": "Climate Change",
        "generated_concept": "Global Warming",
        "explanation": "The long-term increase in Earth's average temperature."
    },
    {
        "concept": "Climate Change",
        "generated_concept": "Carbon Emissions",
        "explanation": "Greenhouse gases released by human activities."
    },
    {
        "concept": "Climate Change",
        "generated_concept": "Renewable Energy",
        "explanation": "Sustainable energy sources reducing climate impact."
    },
    {
        "concept": "Climate Change",
        "generated_concept": "Ocean Acidification",
        "explanation": "The decrease in ocean pH due to CO2 absorption."
    },
    {
        "concept": "Climate Change",
        "generated_concept": "Deforestation",
        "explanation": "The large-scale removal of forests affecting climate balance."
    },
    {
        "concept": "Climate Change",
        "generated_concept": "Environmental Economics",
        "explanation": "The study of economic impacts on climate policies."
    },
    {
        "concept": "Climate Change",
        "generated_concept": "Climate Policy",
        "explanation": "Governmental and global policies addressing climate issues."
    },
    {
        "concept": "Climate Change",
        "generated_concept": "Green Technologies",
        "explanation": "Innovations aimed at reducing environmental impact."
    },
    {
        "concept": "Climate Change",
        "generated_concept": "Geoengineering",
        "explanation": "Large-scale intervention methods to counteract climate change."
    },
    {
        "concept": "Climate Change",
        "generated_concept": "Sustainable Development",
        "explanation": "Balancing economic growth with environmental sustainability."
    }
]


GENERATE_PROMPT = """
You are an expert writer with a deep understanding of linguistic styles and sentiment expression. 
Your task is to generate text based on the given parameters.

**Input Parameters:**
- **ID:** {id}
- **Concept:** {concept}
- **Subconcept:** {extended_concept}
- **Explanation:** {explanation}
- **Writing Style:** {writing_style} 
- **Medium:** {medium}
- **Persona:** {persona}
- **Intention:** {intention}
- **Tone:** {tone}
- **Audience:** {audience}
- **Context:** {context}
- **Language Register:** {language_register} 
- **Sentiment Label:** {sentiment_label} (e.g., Positive, Negative, Neutral, 1-star, 5-star)
- **Sentence Length:** {sentence_length} (e.g., 1 sentence, 2 sentences, short paragraph, long paragraph)

### **Instructions:**
1. Write the text in the **{medium}** format from the perspective of **{persona}**.
2. Maintain the given **{writing_style}** throughout.
3. The text should reflect a **{sentiment_label}** sentiment.
4. The length of the text should match the **{sentence_length}** parameter.

### **Example Output Format:**
- Generated Text: "..."
{format_instructions}

Now, generate the text.
"""

ASPECT_GENERATION_PROMPT = """
You are an expert in ontology design, knowledge representation, and concept hierarchy analysis. 
Your task is to generate aspects for a given concept. These will be used in Aspect-Based Sentiment Analysis (ABSA).
The aspects should be relevant to the concept and as many as wanted.

Concept: {concept}
The number of aspects: {num_aspects}
Aspects:

Format instructions: {format_instructions}

"""

ABSA_GENERATE_PROMPT = """
You are an expert writer with a deep understanding of linguistic styles and sentiment expression. 
Your task is to generate text based on the given parameters. This text will be used for Aspect-Based Sentiment Analysis (ABSA).
Therefore you will create a text with given parameters that reflects the sentimens for each aspect for both respectively.

**Input Parameters:**
- **ID:** {id}
- **Concept:** {concept}
- **Aspects:** {aspects}
- **Writing Style:** {writing_style} 
- **Medium:** {medium}
- **Persona:** {persona}
- **Intention:** {intention}
- **Tone:** {tone}
- **Audience:** {audience}
- **Context:** {context}
- **Language Register:** {language_register} 
- **Sentiment Label:** {sentiment_label} (Each for one aspect, e.g., Positive, Negative, Neutral, 1-star, 5-star)
- **Sentence Length:** {sentence_length} (e.g., 1 sentence, 2 sentences, short paragraph, long paragraph)

### **Instructions:**
1. Write the text in the **{medium}** format from the perspective of **{persona}**.
2. Maintain the given **{writing_style}** throughout.
3. The text should reflect a **{sentiment_label}** sentiment.
4. The length of the text should match the **{sentence_length}** parameter.

### **Example Output Format:**
- Generated Text: "..."
{format_instructions}

Now, generate the text.
"""