from anonymizers.base_anonymizer import Anonymizer


class LocationAnonymizer(Anonymizer):

    anonymization_type = "location"
    location_prepositions = ["in", "at"]

    def __init__(self):
        self.initialize_spacy_model()

    def is_location(self, index: int) -> bool:
        if self.parsed_user_input[index - 1][0].lower() in self.location_prepositions:
            return True
        else:
            return False

    def scrub(self, user_input: str) -> str:
        self.parsed_user_input = self.parse_user_input(user_input)

        if 'PROPN' not in self.all_pos:
            return user_input # For improved speed of implementation

        anonymized_user_input_list = []
        for index, parsed_word in enumerate(self.parsed_user_input):
            word_text = parsed_word[0]
            word_pos = parsed_word[1]

            if index != 0 and word_pos == 'PROPN':
                previous_word = self.parsed_user_input[index - 1]
                if previous_word[2] == 'prep' or prev_word_anonymized:
                    if self.is_location(index) or (self.is_location(index - 1) and prev_word_anonymized):
                        anonymized_word = "[" + self.anonymization_type.upper() + "]"
                        anonymized_user_input_list.append(anonymized_word)
                        prev_word_anonymized = True
                        continue

            anonymized_user_input_list.append(word_text)
            prev_word_anonymized = False

        anonymized_user_input = " ".join(anonymized_user_input_list)
        anonymized_user_input = self.normalize_user_input(anonymized_user_input)
        return anonymized_user_input