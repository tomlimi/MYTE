import json
from collections import defaultdict
import argparse
import os
from tqdm import tqdm
from typing import Union
import logging

class ByteRewriter:

	LEAF ='[LEAF]'

	def __init__(self, rewriting_rules: Union[str, dict[str, str]]):

		if type(rewriting_rules) == str:
			with open(rewriting_rules, "r") as f:
				rewriting_rules = json.load(f)
		elif not type(rewriting_rules) == dict:
			raise ValueError(f"rewriting_rules should be either a path to json file or a dict, got {type(rewriting_rules)}")

		self.hash_tree = self.construct_hash_tree(rewriting_rules)
		revese_revrewriting_rules = {v:k for k,v in rewriting_rules.items()}
		self.reverse_hash_tree = self.construct_hash_tree(revese_revrewriting_rules)

	def add_leaf(self,hash_tree, byte_in_sequence, byte_out_sequence):

		byte_in_list = byte_in_sequence.split(' ')
		byte_out_list = byte_out_sequence.split(' ')

		tree_pointer = hash_tree
		for b in byte_in_list:
			if b not in tree_pointer:
				tree_pointer[b] = {}
			tree_pointer = tree_pointer[b]

		tree_pointer[self.LEAF] = byte_out_list

	def construct_hash_tree(self, rewritting_rules):

		hash_tree = defaultdict(dict)
		for b in (f"{x:02x}" for x in range(256)):
			hash_tree[b][self.LEAF] = [b]

		for in_seequence, out_sequence in rewritting_rules.items():
			self.add_leaf(hash_tree, in_seequence, out_sequence)

		return hash_tree

	def search_hash_tree(self, byte_sequence):

		tree_pointer = self.hash_tree
		for b in byte_sequence:
			if b in tree_pointer:
				tree_pointer = tree_pointer[b]
			else:
				return None

		return tree_pointer[self.LEAF]


	def rewrite_bytes(self, in_bytes, reverse=False):

		out_bytes = []
		b_start = 0
		b_end = 0

		while b_start < len(in_bytes):
			tree_pointer = self.hash_tree if not reverse else self.reverse_hash_tree
			for j in range(b_start, len(in_bytes)):
				b = in_bytes[j]
				if b in tree_pointer:
					tree_pointer = tree_pointer[b]
				elif j == b_start:
					# logging.warning(f"Unrecognized byte {b} in {in_bytes}, Skipping!")
					cur_leaf = [b]
					b_end = j
					break
				else:
					break
				if self.LEAF in tree_pointer:
					cur_leaf = tree_pointer[self.LEAF]
					b_end = j
			out_bytes.extend(cur_leaf)
			b_start = b_end + 1

		return out_bytes


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description="Rewrite bytes")
	parser.add_argument("--byte_map_dir", default="../byte_maps")
	parser.add_argument("--input_dir", default="../morfessor_in")
	parser.add_argument("--output_dir", default="../rewriten")
	parser.add_argument("--languages", nargs="+", default=["eng_Latn", "spa_Latn", "por_Latn", "fra_Latn", "ita_Latn",
	                                                       "ron_Latn", "pol_Latn", "mlt_Latn", "heb_Hebr", "arb_Arab",
	                                                       "jpn_Jpan", "kor_Hang", "tel_Telu", "tam_Taml", "bod_Tibt",
	                                                       "sin_Sinh"])
	parser.add_argument("--split", default="devtest")
	parser.add_argument("--bp", default=48)
	parser.add_argument("--ml", default=0)
	parser.add_argument("--mtn", default=4096)
	parser.add_argument("--reverse", action="store_true", default=False)

	args = parser.parse_args()

	# byte map loading file
	byte_map_file = f"{args.byte_map_dir}/morfessor_bp_{args.bp}_ml_{args.ml}_mtn_{args.mtn}.json"
	rewriter = ByteRewriter(byte_map_file)

	for lang in args.languages:
		# open input corpus file
		corpus_file = f"{args.input_dir}/{lang}/corpus_{args.split}.txt"
		in_lines_bytes = 0.
		out_lines_bytes = 0.
		out_lines = []
		with open(corpus_file, "r") as f:
			in_lines = f.readlines()

		for line in tqdm(in_lines, desc= f"Processing {args.split} corpus in {lang}"):
			in_hex = line.strip().split('-')
			in_lines_bytes += len(in_hex)
			out_hex = rewriter.rewrite_bytes(in_hex, reverse=args.reverse)
			out_lines_bytes += len(out_hex)
			out_lines.append("-".join(out_hex) + '\n')

		print(f"Language: {lang} rewritten from {in_lines_bytes/len(in_lines)} to {out_lines_bytes/len(out_lines)} bytes")

		# open output corpus file
		os.makedirs(f"{args.output_dir}/{lang}", exist_ok=True)
		out_corpus_file = f"{args.output_dir}/{lang}/corpus_{args.split}.txt"
		with open(out_corpus_file, "w") as f:
			f.writelines(out_lines)
