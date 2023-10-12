import tokenizers

from tokenizers import (
	decoders,
	models,
	normalizers,
	pre_tokenizers,
	processors,
	trainers,
	Tokenizer,
)
from transformers import PreTrainedTokenizerFast


def build_gpt_tokenizer(data_path, vocab_sizes):
	vocabulary_dict = {}
	for vocab_size in vocab_sizes:
		tokenizer = Tokenizer(models.BPE())
		tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
		trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"])
		tokenizer.model = models.BPE()
		tokenizer.train([data_path], trainer=trainer)

		tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
		tokenizer.decoder = decoders.ByteLevel()

		wrapped_tokenizer = PreTrainedTokenizerFast(
			tokenizer_object=tokenizer,
			bos_token="<|endoftext|>",
			eos_token="<|endoftext|>", )

		vocabulary_dict[f"tokenizer_{vocab_size}"] = wrapped_tokenizer

	return vocabulary_dict