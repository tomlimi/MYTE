import tokenizers
import tqdm

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

from src.utils import hex_to_str, str_to_hex


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


def build_lexicon_tokenizer(lexicon_iterator, vocab_size, type="bpe"):
	if type == "bpe":
		tokenizer = Tokenizer(models.BPE())
		trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"])
		tokenizer.model = models.BPE()
	elif type == "unigram":
		tokenizer = Tokenizer(models.Unigram())
		trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=["<|endoftext|>"], unk_token="<|endoftext|>")
		tokenizer.model = models.Unigram()

	tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
	tokenizer.train_from_iterator(tqdm.tqdm(lexicon_iterator), trainer=trainer)

	tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
	tokenizer.decoder = decoders.ByteLevel()

	# why is it necessary to wrap the tokenizer? Ask Oreva
	# wrapped_tokenizer = PreTrainedTokenizerFast(
	# 	tokenizer_object=tokenizer,
	# 	bos_token="<|endoftext|>",
	# 	eos_token="<|endoftext|>", )



	return tokenizer

def get_lexeme(lexicon_dir, lang, counted=False, output_str=True):

	counts = []
	lexemes = []
	with open(f"{lexicon_dir}/{lang}_lex.txt") as f:
		lexicon = f.read().splitlines()
	for line in lexicon:
		count, lexeme = line.strip().split("\t")
		if output_str:
			lexeme = hex_to_str(lexeme)
		if counted:
			lexeme = ' '.join([lexeme] * int(count))
		yield lexeme