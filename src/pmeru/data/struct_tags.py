import re


class StructTagger:
    """
    Computes structural depth tags for source code.

    The tag at each character represents the 'Composite Depth':
        Tag = IndentationDepth + BracketNestingDepth

    Example:
        def foo():          # Indent 0, Bracket 0 -> Tag 0
            if (x):         # Indent 1, Bracket 1 (inside parens) -> Tag 2
                return      # Indent 2 -> Tag 2

    This helps the Prime Stream (GRU) track scope even if tokens are lost.
    """

    def __init__(
        self, max_tags=32, indent_size=4, open_brackets="([{", close_brackets=")]}"
    ):
        """
        Args:
            max_tags (int): Maximum tag ID (tags will be clamped to max_tags - 1).
            indent_size (int): Number of spaces that constitute one indentation level.
            open_brackets (str): Characters that increment depth.
            close_brackets (str): Characters that decrement depth.
        """
        self.max_tags = max_tags
        self.indent_size = indent_size
        self.open_set = set(open_brackets)
        self.close_set = set(close_brackets)

    def normalize_tags(self, text: str):
        """
        Generates character-level structural tags.

        Algorithm:
        1. Parse line indentation (spaces/tabs converted to levels).
        2. Walk characters, updating bracket depth.
        3. Sum Indent + Bracket and clamp.
        """
        char_tags = []
        bracket_depth = 0

        lines = text.splitlines(keepends=True)

        for line in lines:
            # 1. Calculate Indentation Level for this line
            leading_spaces = 0
            for char in line:
                if char == " ":
                    leading_spaces += 1
                elif char == "\t":
                    # Assume tab brings us to the next multiple of indent_size
                    # Standard assumption: tabs align to indent_size (often 4)
                    leading_spaces += self.indent_size
                else:
                    break

            indent_level = leading_spaces // self.indent_size

            # 2. Process chars
            for char in line:
                if char in self.open_set:
                    bracket_depth += 1
                elif char in self.close_set:
                    bracket_depth = max(0, bracket_depth - 1)

                # Composite Depth
                total_depth = indent_level + bracket_depth

                # Clamp
                tag = min(total_depth, self.max_tags - 1)
                char_tags.append(tag)

        return char_tags


def align_tags_to_tokens(tokenizer, text, char_tags):
    """
    Aligns character-level tags to token-level tags.
    Strategy: Take the tag of the first character of the token.
    """
    # Tokenize with offsets to map tokens -> chars
    # add_special_tokens=False is important to avoid [CLS]/[BOS] drift if manually handling
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)

    # enc.offset_mapping gives [(start, end), (start, end)...] for each token
    offsets = enc.offset_mapping
    token_tags = []

    # char_tags length should match text length ideally
    # But text might be normalized by tokenizer? Usually 1:1 for offset mapping

    text_len = len(text)
    tags_len = len(char_tags)

    for start, end in offsets:
        # Guard against index out of bounds if tokenizer does something weird
        idx = min(start, tags_len - 1)
        if idx < 0:
            tag = 0
        else:
            tag = char_tags[idx]

        token_tags.append(tag)

    return token_tags
