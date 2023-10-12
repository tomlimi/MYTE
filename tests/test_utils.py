from src.utils import hex_to_str, str_to_hex, bytes_to_hex

import unittest

class TestUtils(unittest.TestCase):

	def test_sth_space(self):

		in_str = "hello world"
		out_hex = "68 65 6c 6c 6f 20 77 6f 72 6c 64"
		self.assertEqual(str_to_hex(in_str), out_hex)

	def test_sth_sep(self):

		in_str = "hello world"
		out_hex = "68-65-6c-6c-6f-20-77-6f-72-6c-64"

		self.assertEqual(str_to_hex(in_str, '-'), out_hex)

	def test_bth(self):

		in_bytes = b'hello world'
		out_hex = "68 65 6c 6c 6f 20 77 6f 72 6c 64"

		self.assertEqual(bytes_to_hex(in_bytes), out_hex)

	def test_hts(self):

		in_hex = "68 65 6c 6c 6f 20 77 6f 72 6c 64"
		out_str = "hello world"

		self.assertEqual(hex_to_str(in_hex), out_str)

	def test_reversibility(self):

		in_str = "hello world"
		out_str = hex_to_str(str_to_hex(in_str))

		self.assertEqual(in_str, out_str)

	def test_reversibility_non_latin(self):

		in_str = "你好世界"
		out_str = hex_to_str(str_to_hex(in_str))

		self.assertEqual(in_str, out_str)

