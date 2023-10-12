import json
from collections import defaultdict


class ByteRewriter:

	LEAF ='[LEAF]'

	def __init__(self, rewritting_rules_file):

		rewriting_rules = {}
		with open(rewritting_rules_file, "r") as f:
			rewriting_rules = json.load(f)

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
		for b in (hex(x)[2:] for x in range(256)):
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
				else:
					break
				if self.LEAF in tree_pointer:
					cur_leaf = tree_pointer[self.LEAF]
					b_end = j
			out_bytes.extend(cur_leaf)
			b_start = b_end + 1

		return out_bytes

if __name__ == "__main__":
	rewriter = ByteRewriter("../byte_maps/simple_decompose.json")
	#print(rewriter.rewrite_bytes("68 65 6c 6c 6f 20 77 6f 72 6c 64".split(' ')))

	rewriter = ByteRewriter("../byte_maps/simple_merge.json")
	print(rewriter.rewrite_bytes("74 6f 6b".split(' ')))