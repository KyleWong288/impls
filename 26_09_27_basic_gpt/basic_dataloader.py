import torch

from collections.abc import Callable
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


# We pack sequences by padding based on batch-wise longest sequence
def collate_bshd(raw_samples: list[str], max_seq_len: int, tokenizer: PreTrainedTokenizerBase) -> tuple[torch.Tensor, torch.Tensor]:
    assert tokenizer.pad_token_id is not None, "tokenizer requires pad token registered"
    assert tokenizer.eos_token_id is not None, "tokenizer requires eos token registered"
    
    # Tokenize and truncate
    encoded = tokenizer(
        raw_samples,
        add_special_tokens=False,
        truncation=True,
        max_length=max_seq_len
    )["input_ids"]
    
    # Add eos
    for sample in encoded:
        sample.append(tokenizer.eos_token_id)

    # Convert list to tensors
    samples = [torch.tensor(sample, dtype=torch.long) for sample in encoded]
    
    # Pad
    batch = torch.nn.utils.rnn.pad_sequence(samples, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    inputs = batch[:, :-1]
    targets = batch[:, 1:]
    return (inputs, targets)


# We pack sequences by packing multiple sequences into one stream
def collate_thd():
    raise NotImplementedError


# Streaming interface that yields a batch of tokenized samples
# External collate_fn() determines packing (i.e., BSHD, THD)
class BasicDataLoader:
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        ds_path: str = "HuggingFaceFW/fineweb",
        ds_name: str = "sample-10BT",
        tokenizer_name: str = "openai-community/gpt2",
        collate_fn: Callable = collate_bshd
    ):
        self.dataset = self._load_dataset(ds_path, ds_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        # Normally, documents are pre-tokenized. For us, collate will handle tokenization
        self.collate_fn = collate_fn

    # Yields a batch of tokenized samples
    def __iter__(self):
        dataset_iter = iter(self.dataset)

        # Outer loop to track iter across next() calls
        while True:
            # Inner loop for a single batch
            batch = []
            for _ in range(self.batch_size):
                try:
                    sample = next(dataset_iter)
                    text = sample["text"]
                except StopIteration:
                    break
                batch.append(text)
            # Yield if batch is non-empty, else signal StopIteration
            if batch:
                inputs, targets = self.collate_fn(batch, self.max_seq_len, self.tokenizer)
                yield (inputs, targets)
            else:
                return

    # Load from hf
    def _load_dataset(self, ds_path: str, ds_name: str, split: str = "train"):
        dataset = load_dataset(
            path=ds_path,
            name=ds_name,
            split=split,
            streaming=True
        )
        return dataset
