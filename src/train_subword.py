import tqdm
import argparse

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

		vocabulary_dict[f"tokenizer_{vocab_size}"] = tokenizer # wrapped_tokenizer

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

	tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
	tokenizer.train_from_iterator(tqdm.tqdm(lexicon_iterator), trainer=trainer)

	tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
	tokenizer.decoder = decoders.ByteLevel()

	# why is it necessary to wrap the tokenizer? Ask Oreva
	wrapped_tokenizer = PreTrainedTokenizerFast(
	 	tokenizer_object=tokenizer,
	 	bos_token="<|endoftext|>",
	 	eos_token="<|endoftext|>", )

	return wrapped_tokenizer


def get_lexeme(lexicon_file, lang, counted=False, output_str=True):

	counts = []
	lexemes = []
	with open(lexicon_file) as f:
		lexicon = f.read().splitlines()
	for line in lexicon:
		count, lexeme = line.strip().split("\t")
		if output_str:
			lexeme = hex_to_str(lexeme)
		if counted:
			lexeme = ' '.join([lexeme] * int(count))
		yield lexeme


if __name__ == "__main__":

	argparser = argparse.ArgumentParser()
	argparser.add_argument("--lexicon")
	argparser.add_argument("--out_dir")
	argparser.add_argument("--language", type=str, required=True)
	argparser.add_argument("--vocab_size", type=int, default=5000)
	argparser.add_argument("--type", default="unigram")
	args = argparser.parse_args()

	lexicon_iterator = get_lexeme(args.lexicon, args.language, counted=False, output_str=True)
	for lexeme in lexicon_iterator:
		print(lexeme)
	lexicon_iterator = get_lexeme(args.lexicon, args.language, counted=False, output_str=True)
	tokenizer = build_lexicon_tokenizer(lexicon_iterator, args.vocab_size, type=args.type)
	tokenizer.save(f"{args.out_dir}/{args.language}_{args.type}_{args.vocab_size}")

