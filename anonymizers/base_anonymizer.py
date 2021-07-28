from typing import List, Tuple
import spacy
import re

parse = List[Tuple[str, str, str]]


class Anonymizer:
    nlp = None
    all_pos = set()

    def __init__(self, Anonymizers):
        for childAnonymizer in Anonymizers:
            if not issubclass(childAnonymizer, Anonymizer):
                raise TypeError("Anonymizer() instance must be initalized "
                                "with list of Anonymizer child classes")
        self.anonymizers = Anonymizers

    def initialize_spacy_model(self):
        """
            Initialize spaCy's Convolutional Neural Network model
            with GloVe vectors trained on Common Crawl data
        """
        if not self.nlp:  # Only initialize spaCy CNN model once
            self.nlp = spacy.load("en_core_web_md")

    def parse_user_input(self, user_input: str) -> parse:
        """
            Takes a user input string and returns a list of tuple triplets,
            with each tuple containing (the initial word, its part of speech,
            its synctactic dependency label)
        """
        parsed_input = []
        spacy_nlp_object = self.nlp(user_input)
        for token in spacy_nlp_object:
            parsed_input.append((token.text, token.pos_, token.dep_))
            self.all_pos.add(token.pos_)
        return parsed_input

    def normalize_user_input(self, user_input: str) -> str:
        replacements = [["\$\s", "$"], ["\(\s", "("], ["\s\)", ")"],
                        ["\s\?", "?"], ["\s\.", "."], ["\s'", "'"], ["\s!", "!"],
                        ["\s(?=[^\w$]\s)", ""], ["(?<=\w)n't(?=[^\w])", " not"]]
        for before, after in replacements:
            user_input = re.sub(before, after, user_input)
        return user_input

    def anonymize_input_data(self, input_data: dict) -> dict:
        output_data = {"examples": []}
        for example in input_data["examples"]:
            anonymized_example = self.anonymization_pipeline(example)
            output_data["examples"].append(anonymized_example)
        return output_data

    def anonymization_pipeline(self, user_input: str) -> str:
        user_input = self.normalize_user_input(user_input)
        for anonymizer in self.anonymizers:
            user_input = anonymizer().scrub(user_input)
        return user_input

    def scrub(self, parsed_input: parse) -> str:
        raise NotImplementedError