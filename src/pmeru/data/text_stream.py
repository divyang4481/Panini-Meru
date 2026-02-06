from torch.utils.data import IterableDataset
from datasets import load_dataset
import torch
import logging
from .struct_tags import StructTagger, align_tags_to_tokens

logger = logging.getLogger(__name__)


class TextStreamDataset(IterableDataset):
    def __init__(
        self,
        tokenizer,
        dataset_name="flytech/python-codes-25k",
        dataset_config=None,
        split="train",
        seq_len=1024,
        skip_batches=0,
        struct_tag_mode="simple",  # simple, none, (future: spacy)
        text_column="text",  # Primary column to look for
        fallback_columns=None,  # List of other columns to try (e.g. ['content', 'code'])
        infinite_loop=False,  # If True, restart dataset when exhausted
        pad_last_batch=False,  # If True, pad the remainder instead of dropping
    ):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.seq_len = seq_len
        self.skip_batches = skip_batches
        self.struct_tag_mode = struct_tag_mode
        self.infinite_loop = infinite_loop
        self.pad_last_batch = pad_last_batch

        self.text_column = text_column
        self.fallback_columns = fallback_columns or ["output", "content", "code"]

        logger.info(f"Initializing TextStreamDataset: {dataset_name} (split={split})")
        logger.info(f"  Seq Len: {seq_len}, Tag Mode: {struct_tag_mode}")
        logger.info(f"  Skip Batches: {skip_batches}, Infinite Loop: {infinite_loop}")

        # Optimize tagger init
        if self.struct_tag_mode == "simple":
            self.tagger = StructTagger()
        else:
            self.tagger = None  # 'none' mode or implementation pending
            logger.info("  StructTagger disabled (mode is not 'simple').")

    def _get_stream(self):
        try:
            ds = load_dataset(
                self.dataset_name, self.dataset_config, split=self.split, streaming=True
            )
            logger.info(f"  Stream loaded successfully for {self.dataset_name}")
            return ds
        except Exception as e:
            logger.error(f"  Failed to load dataset stream: {e}")
            raise e

    def __iter__(self):
        buffer_tokens = []
        buffer_tags = []
        batch_counter = 0

        # Generator loop
        while True:
            dataset_stream = self._get_stream()
            records_processed = 0

            for record in dataset_stream:
                records_processed += 1
                # 1. Extract Text
                text = ""
                # Priority 1: Specified text column
                if self.text_column in record:
                    text = record[self.text_column]
                else:
                    # Priority 2: Fallbacks
                    found = False
                    for col in self.fallback_columns:
                        if col in record:
                            # Special handling for flytech/python-codes-25k which splits instruction/output
                            if (
                                col == "output"
                                and "instruction" in record
                                and record["instruction"]
                            ):
                                text = f"# {record['instruction']}\n{record['output']}"
                            else:
                                text = record[col]
                            found = True
                            break

                    if not found:
                        continue  # Skip bad records

                if not text:
                    continue

                # 2. Structural Tagging
                if self.struct_tag_mode == "simple":
                    try:
                        char_tags = self.tagger.normalize_tags(text)
                        token_tags = align_tags_to_tokens(
                            self.tokenizer, text, char_tags
                        )
                    except Exception as e:
                        logger.warning(
                            f"  Tagging failed for record {records_processed}: {e}.Using zeros."
                        )
                        # Fallback to zeros if tagging fails
                        tokens_temp = self.tokenizer(text, add_special_tokens=False)[
                            "input_ids"
                        ]
                        token_tags = [0] * len(tokens_temp)
                else:
                    # Mode 'none' or fallback
                    pass

                # 3. Tokenize
                tokens = self.tokenizer(text, add_special_tokens=False)["input_ids"]

                if self.struct_tag_mode != "simple":
                    token_tags = [0] * len(tokens)

                buffer_tokens.extend(tokens)
                buffer_tags.extend(token_tags)

                # 4. Yield Batches
                while len(buffer_tokens) >= self.seq_len:
                    # Skipping logic for resumption
                    if batch_counter < self.skip_batches:
                        buffer_tokens = buffer_tokens[self.seq_len :]
                        buffer_tags = buffer_tags[self.seq_len :]
                        batch_counter += 1
                        if batch_counter % 100 == 0:
                            logger.info(f"  Skipped {batch_counter} batches...")
                        continue

                    # Yield Strict Dtypes
                    yield {
                        "input_ids": torch.tensor(
                            buffer_tokens[: self.seq_len], dtype=torch.long
                        ),
                        "attention_mask": torch.ones(
                            self.seq_len, dtype=torch.bool
                        ),  # FlashAttn prefers bool or int
                        "struct_tags": torch.tensor(
                            buffer_tags[: self.seq_len], dtype=torch.long
                        ),
                        "labels": torch.tensor(
                            buffer_tokens[: self.seq_len], dtype=torch.long
                        ),
                    }

                    buffer_tokens = buffer_tokens[self.seq_len :]
                    buffer_tags = buffer_tags[self.seq_len :]
                    batch_counter += 1

            # End of Dataset
            logger.info(
                f"  Dataset stream exhausted. Processed {records_processed} records."
            )

            if (
                self.pad_last_batch
                and len(buffer_tokens) > 0
                and batch_counter >= self.skip_batches
            ):
                # Pad remainder
                pad_len = self.seq_len - len(buffer_tokens)
                pad_token = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

                padded_tokens = buffer_tokens + [pad_token] * pad_len
                padded_tags = buffer_tags + [0] * pad_len
                mask = [1] * len(buffer_tokens) + [0] * pad_len

                yield {
                    "input_ids": torch.tensor(padded_tokens, dtype=torch.long),
                    "attention_mask": torch.tensor(mask, dtype=torch.bool),
                    "struct_tags": torch.tensor(padded_tags, dtype=torch.long),
                    "labels": torch.tensor(padded_tokens, dtype=torch.long),
                }
                batch_counter += 1

            if not self.infinite_loop:
                logger.info("  Standard iteration finished. Stopping.")
                break
            else:
                logger.info("  Infinite loop enabled. Restarting stream.")
