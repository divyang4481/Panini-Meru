import unittest
from src.pmeru.data.struct_tags import StructTagger


class TestStructTagger(unittest.TestCase):
    def setUp(self):
        self.tagger = StructTagger(indent_size=4, max_tags=10)

    def test_indentation_simple(self):
        code = "def foo():\n    pass"
        # Line 1: indent 0.
        # Line 2: indent 4 spaces -> level 1.
        tags = self.tagger.normalize_tags(code)

        # 'd' (0) 'e' (0) ... 'Pass' should be level 1.
        # Find index of 'p' in 'pass'
        idx_pass = code.find("pass")
        self.assertEqual(tags[idx_pass], 1, "Pass should have tag 1 (indent 1)")
        self.assertEqual(tags[0], 0, "Def should have tag 0")

    def test_brackets(self):
        code = "x = (a + b)"
        # x(0) = (0) space(0) ((1) a(1) ... )(0)
        tags = self.tagger.normalize_tags(code)
        idx_open = code.find("(")
        idx_a = code.find("a")

        self.assertEqual(
            tags[idx_open],
            1,
            "Open paren itself should increment depth (or typically be 1?)",
        )
        # In current impl: char in "([{" -> depth += 1.
        # So the '(' gets depth 1. 'a' gets depth 1.

        self.assertEqual(tags[idx_a], 1, "Content inside paren should be depth 1")

    def test_mixed(self):
        code = "    if (True):"
        # Indent 1.
        # ( -> Indent 1 + Bracket 1 = 2
        tags = self.tagger.normalize_tags(code)
        idx_t = code.find("True")
        self.assertEqual(
            tags[idx_t], 2, "Inside indented parenthesis should be depth 2"
        )

    def test_custom_indent(self):
        tagger_2 = StructTagger(indent_size=2)
        code = "  return"
        tags = tagger_2.normalize_tags(code)
        idx_r = code.find("return")
        self.assertEqual(
            tags[idx_r], 1, "2 spaces should be indent 1 with custom setting"
        )


class TestTextStreamDataset(unittest.TestCase):
    def setUp(self):
        # Mock tokenizer
        class MockTokenizer:
            pad_token_id = 0
            eos_token_id = 0

            def __call__(self, text, add_special_tokens=False):
                # Simple mock: 1 token per char
                return {"input_ids": [1] * len(text)}

        self.tokenizer = MockTokenizer()

    def test_indent_propagation(self):
        # Ensure indent_size passed to dataset reaches the tagger
        from src.pmeru.data.text_stream import TextStreamDataset

        # 1. Init with indent_size=2
        ds = TextStreamDataset(
            self.tokenizer,
            dataset_name=None,
            split="train",
            struct_tag_mode="simple",
            indent_size=2,
        )
        # Check internal tagger
        self.assertEqual(
            ds.tagger.indent_size, 2, "Dataset should pass indent_size=2 to tagger"
        )

        # 2. Init with indent_size=4
        ds4 = TextStreamDataset(
            self.tokenizer,
            dataset_name=None,
            split="train",
            struct_tag_mode="simple",
            indent_size=4,
        )
        self.assertEqual(
            ds4.tagger.indent_size, 4, "Dataset should pass indent_size=4 to tagger"
        )


if __name__ == "__main__":
    unittest.main()
