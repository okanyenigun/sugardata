def get_ner_localization_prompt():
    return """
You will be given:
1) A reference text in some source language.
2) A mapping of one or more tokens in that text to their entity category (e.g., person, organization, location).
3) A target language.

# Primary Objective (strict):
Translate/localize the reference text into the target language AND for every key in Target word mapping:
- REPLACE that exact token in the text with a SPECIFIC, culturally/linguistically appropriate equivalent in the target language (NOT a generic noun).
- The chosen replacement MUST appear verbatim in the localized text.
- The original token MUST NOT appear ANYWHERE in the final output (no parentheses, no footnotes, no acronyms).

# Replacement rules:
- Prefer a real, well-known equivalent in the target language/locale. If no clear real-world equivalent exists, invent a plausible, natural-sounding name consistent with the category.
- Do not output generic category words (e.g., "organization", "kuruluş", "company", "şehir"). Always pick a concrete, proper-noun style replacement.
- Keep the rest of the text faithful, but correctness of the entity substitution is the priority.

# Examples (few-shot):

Example A (organization):
- Reference text: "Pakistani scientists and engineers working at IAEA became aware of advancing Indian nuclear program towards making the bombs."
- Target word mapping: {{"IAEA": "organization"}} 
- Target language: Turkish
- Good replacement choice: "Türkiye Atom Enerjisi Kurumu" (concrete, localized); BAD: "kuruluş" (generic).
- Expected behavior: The localized text contains "...Türkiye Atom Enerjisi Kurumu..." and does NOT contain "IAEA".

Example B (location):
- Mapping: {{"Oxford": "location"}} → In French, use a specific French-relevant equivalent or a plausible French locale name if context demands a localized stand-in, not "ville".

Example C (person):
- Mapping: {{"John Smith": "person"}} → In Japanese, use a realistic Japanese full name (e.g., "佐藤 健") rather than "人".

Now perform the task.
Index: {index}
Reference text: {reference_text}
Target word mapping: {target_word_mapping}
Target language: {target_language}

Format Instructions:
{format_instructions}
"""
