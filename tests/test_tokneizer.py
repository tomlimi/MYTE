
import cProfile

import unittest
import time

from myt5.myt5_tokenizer import MyT5Tokenizer
from transformers import AutoTokenizer


class TestTokenizer(unittest.TestCase):

	tokenizer = MyT5Tokenizer()
	ref_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", use_fast=False)

	def test_simple_tokenize(self):

		in_str = "Hello World"
		out_tokens = ['52', '85', '91', '9f', '6f', '20', '52', '85', '9f', '90']

		self.assertEqual(self.tokenizer.tokenize(in_str), out_tokens)

		in_pl_str = "Witaj świecie"
		out_tokens = ['77', '41', '69', '74', '61', '6a', '20', '4b', 'a5', '97', '63', '69', '65']

		self.assertEqual(self.tokenizer.tokenize(in_pl_str), out_tokens)

		in_jp_str = "こんにちは世界"
		out_tokens = ['58', '80', '91', 'a1', 'e4', 'b8', '96', 'e7', '95', '8c']

		self.assertEqual(self.tokenizer.tokenize(in_jp_str), out_tokens)

	def test_batch_tokenize(self):

		in_batch = ["Hello World", "Witaj świecie", "こんにちは世界"]

		out_tokens = [['52', '85', '91', '9f', '6f', '20', '52', '85', '9f', '90', '</s>'],
					['77', '41', '69', '74', '61', '6a', '20', '4b', 'a5', '97', '63', '69', '65', '</s>'],
					['58', '80', '91', 'a1', 'e4', 'b8', '96', 'e7', '95', '8c', '</s>']]

		self.assertListEqual(
			[self.tokenizer.convert_ids_to_tokens(ids) for ids in  self.tokenizer(in_batch)["input_ids"]],
			out_tokens)

	def test_time_simple_tokenizer(self):
		profiler = cProfile.Profile()
		ko_string = "이것은 테스트입니다" * 10000

		start_ref = time.time()
		self.ref_tokenizer.tokenize(ko_string)
		end_ref = time.time()

		start = time.time()
		self.tokenizer.tokenize(ko_string)
		end = time.time()

		print(f"Reference tokenizer took {end_ref - start_ref} seconds")
		print(f"Tested tokenizer took {end - start} seconds")
		print(f"Slowdown factor: {(end - start) / (end_ref - start_ref)}")
		self.assertLess(end - start, (end_ref - start_ref) * 6.)

	def test_time_batch_tokenizer(self):
		profiler = cProfile.Profile()
		for i in range(5):
			ko_strings = ["이것은 테스트입니다" * (10 **(4 - i))] * (10 ** i)

			start_ref = time.time()
			self.ref_tokenizer(ko_strings)
			end_ref = time.time()

			start = time.time()
			self.tokenizer(ko_strings)
			end = time.time()

			print(f"Batach size: {10 ** i}")
			print(f"Reference tokenizer took {end_ref - start_ref} seconds")
			print(f"Tested tokenizer took {end - start} seconds")
			print(f"Slowdown factor: {(end - start) / (end_ref - start_ref)}")
			self.assertLess(end - start, (end_ref - start_ref) * 6.)
