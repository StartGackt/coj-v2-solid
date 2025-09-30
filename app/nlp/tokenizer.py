class Tokenizer:
    """
    A class responsible for tokenizing text data.
    This adheres to the Single Responsibility Principle by focusing solely on tokenization logic.
    """

    def __init__(self, tokenizer_function):
        """
        Initialize the tokenizer with a specific tokenization function.

        :param tokenizer_function: A callable that performs tokenization.
        """
        self.tokenizer_function = tokenizer_function

    def tokenize(self, text):
        """
        Tokenize the given text using the provided tokenization function.

        :param text: The text to tokenize.
        :return: A list of tokens.
        """
        return self.tokenizer_function(text)
