

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

STYLE_PREFIX = """
You are an expert in linguistics, discourse analysis, and narrative structures, specializing in identifying writing styles and personas based on context. Your task is to analyze the given concept and generate 10 distinct sentence styles, along with a suitable persona and writing style for each.

For given concept, determine:

10 different writing styles or discourse formats where this concept commonly appears.
A persona (author perspective) that would naturally use this style.
A medium where the sentence will be taken and writing style description, which may include tone, formality, and influences (e.g., technical, poetic, first-person, etc.).
"""

STYLE_SUFFIX = """

Identify 10 possible sentence styles for concept across different domains.
For each style, include:
A title of medium (e.g., "Scientific Research Paper")
A persona (who is writing this?)
A writing style description (tone, structure, level of formality, influences, etc.)
Ensure a mix of formal, informal, technical, creative, and persuasive formats.

Concept: <Insert Given Concept>
1. **Sentence Style**: <Type of text>
   - **Persona**: <Who is writing this?>
   - **Writing Style**: <Description of style, tone, and structure>
2. **Sentence Style**: <Type of text>
   - **Persona**: <Who is writing this?>
   - **Writing Style**: <Description of style, tone, and structure>
...
10. **Sentence Style**: <Type of text>
   - **Persona**: <Who is writing this?>
   - **Writing Style**: <Description of style, tone, and structure>

Format Instructions:
{format_instructions}

The given concept is {concept}
"""

STYLE_EXAMPLE_TEMPLATE = """
Concept: {concept}\nMedium: {medium}\nPersona: {persona}\nWriting Style: {writing_style}
"""


STYLE_EXAMPLES = [
    {
        "concept": "Artificial Intelligence",
        "medium": "Scientific Research Paper",
        "persona": "A Silicon Valley research scientist at an AI lab like OpenAI or DeepMind.",
        "writing_style": "Formal, highly technical, full of citations, structured in abstract, methodology, results, discussion."
    },
    {
        "concept": "Artificial Intelligence",
        "medium": "Business Pitch Deck",
        "persona": "A startup founder at Y Combinator pitching an AI-driven SaaS product.",
        "writing_style": "Persuasive, concise, data-backed, with a focus on market potential and ROI."
    },
    {
        "concept": "Artificial Intelligence",
        "medium": "Fictional Narrative",
        "persona": "A sci-fi author inspired by Isaac Asimov writing a near-future AI-driven dystopia.",
        "writing_style": "Descriptive, world-building, with thought-provoking philosophical undertones."
    },
    {
        "concept": "Artificial Intelligence",
        "medium": "News Headline & Report",
        "persona": "A tech journalist at WIRED or The Verge covering AI breakthroughs.",
        "writing_style": "Catchy, engaging, slightly sensationalized, with a balance of facts and opinion."
    },
    {
        "concept": "Artificial Intelligence",
        "medium": "First-Person Blog Post",
        "persona": "A developer who built an AI chatbot from scratch sharing their experience.",
        "writing_style": "Casual, personal, slightly technical but approachable for non-experts."
    },
    {
        "concept": "Climate Change",
        "medium": "Scientific Journal Article",
        "persona": "A climate scientist at NASA publishing in Nature Climate Change.",
        "writing_style": "Data-heavy, peer-reviewed, objective, structured in sections."
    },
    {
        "concept": "Climate Change",
        "medium": "Government Policy Statement",
        "persona": "A minister from the European Union addressing climate policy.",
        "writing_style": "Formal, structured, politically neutral but decisive."
    },
    {
        "concept": "Climate Change",
        "medium": "Activist Speech",
        "persona": "A Greenpeace activist rallying a protest crowd.",
        "writing_style": "Passionate, urgent, call to action, emotionally engaging."
    },
    {
        "concept": "Climate Change",
        "medium": "News Report",
        "persona": "A BBC environment journalist covering rising sea levels.",
        "writing_style": "Balanced, informative, neutral but engaging."
    },
    {
        "concept": "Climate Change",
        "medium": "Advertising Campaign",
        "persona": "A marketing strategist for Patagonia creating an ad campaign.",
        "writing_style": "Short, emotional, impactful, visual-driven."
    }
]

GENERATE_PROMPT = """
You are an expert writer with a deep understanding of linguistic styles and sentiment expression. 
Your task is to generate text based on the given parameters.

**Input Parameters:**
- **ID:** {id}
- **Concept:** {concept}
- **Subconcept:** {generated_concept}
- **Explanation:** {explanation}
- **Medium:** {medium} (e.g., news article, academic paper, social media post, fiction, advertisement, etc.)
- **Persona:** {persona} (The author perspective writing this text)
- **Writing Style:** {writing_style} (e.g., formal, technical, emotional, poetic, first-person, etc.)
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
