from src.rewrite_bytes import ByteRewriter
from src.utils import hex_to_str, str_to_hex, bytes_to_hex

import unittest
from tqdm import tqdm

class TestByteRewriterDecompose(unittest.TestCase):

	rewriter = ByteRewriter("../byte_maps/simple_decompose.json")

	def test_simple_decompose(self):
		# test rewriting
		in_str = "Hello WorlD"
		out_str = "hAello wAorldA"

		in_hex = str_to_hex(in_str).split(' ')
		out_hex = str_to_hex(out_str).split(' ')

		self.assertEqual(self.rewriter.rewrite_bytes(in_hex), out_hex)

	def test_simple_decompose_reversible(self):

		in_str = "Hello WorlD"
		out_str = "Hello WorlD"

		in_hex = str_to_hex(in_str).split(' ')
		out_hex = str_to_hex(out_str).split(' ')

		self.assertEqual(self.rewriter.rewrite_bytes(self.rewriter.rewrite_bytes(in_hex), reverse=True), out_hex)

	def test_simple_decompose_non_latin(self):

		rewriter = ByteRewriter("../byte_maps/simple_decompose.json")

		in_str = "你好世界 Hello WorlD"
		out_str = "你好世界 hAello wAorldA"

		in_hex = str_to_hex(in_str).split(' ')
		out_hex = str_to_hex(out_str).split(' ')

		self.assertEqual(self.rewriter.rewrite_bytes(in_hex), out_hex)


class TestByteRewriterMerge(unittest.TestCase):

	rewriter = ByteRewriter("../byte_maps/simple_merge.json")

	def test_simple_merge(self):
		in_str = "hello world tok"
		out_str = "hello world B0"

		in_hex = str_to_hex(in_str).split(' ')
		out_hex = str_to_hex(out_str).split(' ')

		self.assertEqual(self.rewriter.rewrite_bytes(in_hex), out_hex)

	def test_simple_merge_caps(self):

		in_str = "Hello WorlD TOK"
		out_str = "hAello wAorldA B1"

		in_hex = str_to_hex(in_str).split(' ')
		out_hex = str_to_hex(out_str).split(' ')

		self.assertEqual(self.rewriter.rewrite_bytes(in_hex), out_hex)

	def test_simple_merge_reversible(self):

		in_str = "Hello WorlD TOK"
		out_str = "Hello WorlD TOK"

		in_hex = str_to_hex(in_str).split(' ')
		out_hex = str_to_hex(out_str).split(' ')

		self.assertEqual(self.rewriter.rewrite_bytes(self.rewriter.rewrite_bytes(in_hex), reverse=True), out_hex)

	def test_simple_merge_non_latin(self):

		in_str = "你好世界 hello world tok"
		out_str = "你好世界 hello world B0"

		in_hex = str_to_hex(in_str).split(' ')
		out_hex = str_to_hex(out_str).split(' ')

		self.assertEqual(self.rewriter.rewrite_bytes(in_hex), out_hex)


class TestByteRewriterCorpus(unittest.TestCase):

	rewriter = ByteRewriter("../byte_maps/simple_merge.json")
	corpus_path = "joint.dev"

	def test_corpus(self):

		with open(self.corpus_path, "r") as corpus_file:
			lines = corpus_file.readlines()

		for line in tqdm(lines):
			in_hex = str_to_hex(line.strip()).split(' ')
			out_hex = self.rewriter.rewrite_bytes(in_hex)
			self.assertEqual(self.rewriter.rewrite_bytes(out_hex, reverse=True), in_hex)
			if 'tok' in line or 'TOK' in line:
				print(hex_to_str(' '.join(in_hex)))
				print(hex_to_str(' '.join(out_hex)))
