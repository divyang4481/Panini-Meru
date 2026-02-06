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
        self.assertEqual(tags[0], 3, "High risk should be tag 3")

        evt_low = WorkflowEvent(step_id="l1", action="INFO", risk_score=10)
        tags_low = evt_low.to_struct_vector()
        self.assertEqual(tags_low[0], 0, "Low risk should be tag 0")

    def test_tokenizer_shape(self):
        evt = WorkflowEvent(step_id="s1", action="TEST", risk_score=55)
        # Risk 55 -> Tag 2
        input_ids, token_tags = self.evt_tokenizer.encode_event(evt)

        self.assertEqual(len(input_ids), len(token_tags), "Tags and tokens must align")
        self.assertEqual(token_tags[0], 2, "Tag should reflect risk level 2")


if __name__ == "__main__":
    unittest.main()
