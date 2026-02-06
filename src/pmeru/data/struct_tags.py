import re


class StructTagger:
    def __init__(self, max_tags=32):
        self.max_tags = max_tags
        # We reserve tags 0-31 for depth.
        # Tag 0: Top level
        # Tag N: Depth N

    def normalize_tags(self, text: str):
        """
        Generates character-level structural tags based on:
        1. Line Indentation (Python style: 4 spaces = 1 level)
        2. Bracket Nesting (([{...}]))

        Tag[i] = IndentLevel(line) + BracketDepth(char_i)
        clamped to max_tags-1.
        """
        char_tags = []
        bracket_depth = 0

        lines = text.splitlines(keepends=True)

        for line in lines:
            # 1. Calculate Indentation Level for this line
            # Count leading spaces
            leading_spaces = 0
            for char in line:
                if char == " ":
                    leading_spaces += 1
                elif char == "\t":
                    leading_spaces += 4  # Assume tab=4 spaces
                else:
                    break

            indent_level = leading_spaces // 4

            # 2. Process chars
            for char in line:
                if char in "([{":
                    bracket_depth += 1
                elif char in ")]}":
                    bracket_depth = max(0, bracket_depth - 1)

                # Composite Depth
                # We want the tag to reflect "Where am I in the structure?"
                # Total Depth = Indent + Brackets
                total_depth = indent_level + bracket_depth

                # Clamp
                tag = min(total_depth, self.max_tags - 1)
                char_tags.append(tag)

        # Handle case where splitlines missed the trailing EOF if no newline?
        # Python splitlines(keepends=True) usually handles it well.

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
