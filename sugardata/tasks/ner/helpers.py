from typing import List, Dict, Any, Tuple
from .errors import NERValidationError


def validate_localize_ner_input_examples(examples: List[Dict[str, Any]]):
    """
    Example format is strict.
    The format must be like:
    [
        {
            "text": "Barack Obama was born in Hawaii.",
            'ner_tags': [{"Barack Obama": "PERSON"}, {"Hawaii": "LOCATION"}]
        },
        ...
    ]
    """
    example_format_template = """
    [
        {
            "text": "Barack Obama was born in Hawaii.",
            'ner_tags': [{"Barack Obama": "PERSON"}, {"Hawaii": "LOCATION"}]
        },
        ...
    ]
        """
    
    # Validate the type of examples
    if not isinstance(examples, list):
        raise NERValidationError("Input examples must be a list of dictionaries. Example format:\n" + example_format_template)
    for example in examples:
        if not isinstance(example, dict):
            raise NERValidationError("Each example must be a dictionary. Example format:\n" + example_format_template)
        if 'text' not in example or 'ner_tags' not in example:
            raise NERValidationError("Each example must contain 'text' and 'ner_tags' keys. Example format:\n" + example_format_template)
        if not isinstance(example['text'], str):
            raise NERValidationError("'text' must be a string. Example format:\n" + example_format_template)
        if not isinstance(example['ner_tags'], list):
            raise NERValidationError("'ner_tags' must be a list of dictionaries. Example format:\n" + example_format_template)
        for tag in example['ner_tags']:
            if not isinstance(tag, dict) or len(tag) != 1:
                raise NERValidationError("Each item in 'ner_tags' must be a dictionary with a single key-value pair. Example format:\n" + example_format_template)


def validate_entity_labels(entity_labels: Dict[str, Tuple[int, int]]):
    """
    Validate the entity_labels format.
    The format must be like:
    {
        "PERSON": (1, 2),
        "ORG": (3, 4),
        "LOC": (5, 6)
    }
    """
    example_format_template = """
    {
        "PERSON": (1, 2),
        "ORG": (3, 4),
        "LOC": (5, 6)
    }
    """
    if not isinstance(entity_labels, dict):
        raise NERValidationError("Entity labels must be a dictionary. Example format:\n" + example_format_template)
    for key, value in entity_labels.items():
        if not isinstance(key, str):
            raise NERValidationError("Keys in entity_labels must be strings. Example format:\n" + example_format_template)
        if not isinstance(value, tuple) or len(value) != 2:
            raise NERValidationError("Values in entity_labels must be tuples of (start, end). Example format:\n" + example_format_template)
        if not all(isinstance(i, int) for i in value):
            raise NERValidationError("Start and end positions must be integers. Example format:\n" + example_format_template)


def assign_entity_labels(entity_list: List[str]) -> Dict[str, Tuple[int, int]]:
    """
    Assign entity labels to a list of entities.
    Starts from 1 for the first entity.

    Args:
        entity_list (List[str]): A list of entity strings. Example: ["PERSON", "ORG", "LOC"]

    Returns:
        Dict[str, Tuple[int, int]]: A dictionary mapping entity labels to their start and end positions. Example: {
            "PERSON": (1, 2),
            "ORG": (3, 4),
            "LOC": (5, 6)
        }
    """
    return {
        entity: (i * 2 + 1, i * 2 + 2)
        for i, entity in enumerate(entity_list, start=1)
    }   
