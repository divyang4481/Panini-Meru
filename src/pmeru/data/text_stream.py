from torch.utils.data import IterableDataset
from datasets import load_dataset
import torch
from .struct_tags import StructTagger, align_tags_to_tokens


class TextStreamDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        dataset_name="flytech/python-codes-25k",
        dataset_config=None,
        split="train",
        seq_len=1024,
        skip_batches=0,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset = load_dataset(
            dataset_name, dataset_config, split=split, streaming=True
        )
        self.tagger = StructTagger()
        self.skip_batches = skip_batches

    def __iter__(self):
        buffer_tokens = []
        buffer_tags = []

        batch_counter = 0

        for record in self.dataset:
            # Handle different dataset column structures
            text = ""
            if "text" in record:
                text = record["text"]
            elif "output" in record:  # flytech/python-codes-25k
                # Combine instruction and output for full context
                if "instruction" in record and record["instruction"]:
                    text = f"# {record['instruction']}\n{record['output']}"
                else:
                    text = record["output"]
            elif "content" in record:
                text = record["content"]

            if not text:
                continue

            # 1. Generate tags for the text
            char_tags = self.tagger.normalize_tags(text)

            # 2. Tokenize and align
            token_tags = align_tags_to_tokens(self.tokenizer, text, char_tags)
            tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]

            buffer_tokens.extend(tokens)
            buffer_tags.extend(token_tags)

            # Yield full sequences
            while len(buffer_tokens) >= self.seq_len:
                # Check if we need to skip this batch to resume training
                if batch_counter < self.skip_batches:
                    # Skip
                    buffer_tokens = buffer_tokens[self.seq_len :]
                    buffer_tags = buffer_tags[self.seq_len :]
                    batch_counter += 1
                    continue

                # Yield
                yield {
                    "input_ids": torch.tensor(buffer_tokens[: self.seq_len]),
                    "attention_mask": torch.ones(self.seq_len),  # Dense mask
                    "struct_tags": torch.tensor(buffer_tags[: self.seq_len]),
                    "labels": torch.tensor(
                        buffer_tokens[: self.seq_len]
                    ),  # Auto-regressive
                }

                buffer_tokens = buffer_tokens[self.seq_len :]
                buffer_tags = buffer_tags[self.seq_len :]
                batch_counter += 1
