import argparse
import codecs
import re
from collections import defaultdict
from datasets import load_dataset
from datasets.arrow_reader import DatasetNotOnHfGcsError
from mosestokenizer import MosesTokenizer
import psutil
from torch.utils.data import DataLoader


from rewrite_bytes import ByteRewriter
from utils import hex_to_str, str_to_hex, bytes_to_hex
from tqdm import tqdm


def save_lexicon(lex_counted, language, lex_directory):

	# if lex_directory is not None:
	with codecs.open(f"{lex_directory}/{language}_lex.txt", "w") as lexicon_file:
		for b_lex, count in lex_counted.items():
			lexicon_file.write(f"{count}\t{b_lex}\n")
	print(f"Lexicon saved to {lex_directory}/{language}_lex.txt")


def count_in_corpus(language: str, lexeme_count: dict[str, int], rewriter: ByteRewriter, no_lexicon: bool, corpus: str) -> dict[str, int]:

	lexeme_codes = {lex: f"lex_{lid}" for lid, lex in enumerate(lexeme_count.keys())}
	reverse_lexeme_codes = {v:k for k,v in lexeme_codes.items()}
	lexeme_rewriter = ByteRewriter(lexeme_codes)

	tokenizer = MosesTokenizer(language)
	
	batch_size = 5
	max_dataset_size = 2500000

	if corpus == 'wikipedia':
		try:
			dataset = load_dataset('wikipedia', f"20220301.{language}", split='train', streaming=True)
		except (ValueError, DatasetNotOnHfGcsError):
			print("DirectRunner dataset loaded. OOM may occur! Date:20220301")
			beamed_dataset = load_dataset('wikipedia', date="20230920", language=language, split='train', beam_runner='DirectRunner')
			dataset = beamed_dataset.to_iterable_dataset()
		else:
			print("Streaming wikipedia from HF. Date:20230920")
		dataset = dataset.take(max_dataset_size)
	else:
		raise ValueError(f"Only Wikipedia supported")
	
	def process_wikipedia_example(batch):
		partial_lexeme_count = defaultdict(int)
		#for example in batch["text"]:
		example = "\n".join(batch["text"])
		tokenized_txt = tokenizer(example.replace("\n", " "))
		bytes_normalized = rewriter.rewrite_bytes(str_to_hex(" ".join(tokenized_txt)).split(' '))
		bytes_lexemized = lexeme_rewriter.rewrite_bytes(bytes_normalized)
		# find lexem ids in the text
		lexem_ids = [tok for tok in bytes_lexemized if tok.startswith("lex_")]

		for lid in lexem_ids:
			partial_lexeme_count[reverse_lexeme_codes[lid]] += 1
		batch['lexeme_count'] = [[(lex, count) for lex, count in partial_lexeme_count.items()]]
		return batch

	def process_wikipedia_example_no_lexion(batch):
		partial_lexeme_count = defaultdict(int)

		example = "\n".join(batch["text"])
		tokenized_txt = tokenizer(example.replace("\n", " "))
		for token in tokenized_txt:
			token_normalized = rewriter.rewrite_bytes(str_to_hex(token).split(' '))
			partial_lexeme_count[" ".join(token_normalized)] += 1
		batch['lexeme_count'] = [[(lex, count) for lex, count in partial_lexeme_count.items()]]
		return batch
	
		
	if no_lexicon:
		dataset = dataset.map(lambda x: process_wikipedia_example_no_lexion(x), batched=True, batch_size=batch_size, remove_columns=["text", "title", "url", "id"])
	else:
		dataset = dataset.map(lambda x: process_wikipedia_example(x), batched=True, batch_size=batch_size, remove_columns=["text", "title", "url", "id"])
	
	for batch in tqdm(dataset, desc="Processing Wikipedia lexems"):
		for lexeme, count in batch['lexeme_count']:
			lexeme_count[lexeme] += count

	return lexeme_count


if __name__ == "__main__":

	argparser = argparse.ArgumentParser()
	argparser.add_argument("--language", help="language code to process", default="ro")
	argparser.add_argument("--lexicon_directory", help="directory with directory and lexicon files", default="../lexicons")
	argparser.add_argument("--pre_processing_file", help="file with processing parameters", default="../byte_maps/decompose.json")
	argparser.add_argument("--do_capitalize", help="if capitalize lexemes", action='store_true', default=False)
	argparser.add_argument("--no_lexicon", help="do not save lexicon", action="store_true", default=False)
	argparser.add_argument("--min_occurances", help="minimum occuracnes for a lexeme", default=0, type=int)
	argparser.add_argument("--lexicon_size", help="Pre-set lexicon size", default=50000, type=int)

	args = argparser.parse_args()

	# word counts and save as lexicon
	lexeme_counts = defaultdict(int)
	pp_rewriter = ByteRewriter(args.pre_processing_file)

	no_lexicon = args.no_lexicon
	if not args.no_lexicon and args.lexicon_directory is not None:
		try:
			with open(f"{args.lexicon_directory}/{args.language}_dir.txt", "r") as dictionary_file:
				dictionary_lines = dictionary_file.readlines()
				for line in dictionary_lines:
					_, lexeme = line.split()
					lexeme_counts[lexeme] = 0
					if args.do_capitalize:
						lexeme_counts[lexeme.capitalize()] = 0
		except FileNotFoundError:
			print(f"Warning: {args.language}_dir.txt not found, creating new lexicon")
			no_lexicon = True

	# hexify and normalize lexemes
	if not no_lexicon:
		lexeme_counts = {" ".join(pp_rewriter.rewrite_bytes(str_to_hex(lexeme).split(' '))): count for lexeme, count in tqdm(lexeme_counts.items(), desc="Normalizing lexemes")}
	lexeme_counts = count_in_corpus(args.language, lexeme_counts, pp_rewriter, no_lexicon=no_lexicon, corpus='wikipedia')
	# sort and filter lexemes
	lexeme_counts = {lexeme: count for lexeme, count in lexeme_counts.items() if count >= args.min_occurances}
	if args.lexicon_size is not None:
		lexeme_counts = {lexeme: count for lexeme, count in sorted(lexeme_counts.items(), key=lambda x: x[1], reverse=True)[:args.lexicon_size]}
	else:
		lexeme_counts = {lexeme: count for lexeme, count in sorted(lexeme_counts.items(), key=lambda x: x[1], reverse=True)}

	save_lexicon(lexeme_counts, args.language, args.lexicon_directory)
