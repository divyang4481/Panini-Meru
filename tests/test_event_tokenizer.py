import unittest
from transformers import AutoTokenizer
from src.pmeru.data.event_tokenizer import WorkflowEvent, EventTokenizer


class TestEventTokenizer(unittest.TestCase):
    def setUp(self):
        # Use a lightweight tokenizer for testing
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.evt_tokenizer = EventTokenizer(self.tokenizer)

    def test_event_serialization(self):
        evt = WorkflowEvent(step_id="s1", action="TEST", risk_score=50)
        line = evt.to_text_line()
        self.assertIn("[s1]", line)
        self.assertIn("TEST", line)
        self.assertIn("(Risk: 50)", line)

    def test_risk_tagging(self):
        evt_high = WorkflowEvent(step_id="h1", action="ALERT", risk_score=90)
        tags = evt_high.to_struct_vector()
        # Risk > 80 -> Tag 3
        # Composite = 30 + (hash("ALERT") % 10)
        expected_high = 30 + (hash("ALERT") % 10)
        self.assertEqual(tags[0], expected_high, "High risk composite tag mismatch")

        evt_low = WorkflowEvent(step_id="l1", action="INFO", risk_score=10)
        tags_low = evt_low.to_struct_vector()
        # Risk < 20 -> Tag 0
        # Composite = 0 + (hash("INFO") % 10)
        expected_low = hash("INFO") % 10
        self.assertEqual(tags_low[0], expected_low, "Low risk composite tag mismatch")

    def test_tokenizer_shape(self):
        evt = WorkflowEvent(step_id="s1", action="TEST", risk_score=55)
        # Risk 55 -> Tag 2
        input_ids, token_tags = self.evt_tokenizer.encode_event(evt)

        self.assertEqual(len(input_ids), len(token_tags), "Tags and tokens must align")

        expected_test = 20 + (hash("TEST") % 10)
        self.assertEqual(
            token_tags[0], expected_test, "Tag should reflect risk level 2 + action"
        )


if __name__ == "__main__":
    unittest.main()
