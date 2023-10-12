import codecs
import argparse
import binascii


def str_to_hex(line: str, sep: str = ' ') -> str:
	return bytes_to_hex(bytes(line, 'utf-8'), sep)

def bytes_to_hex(bline: str, sep: str = ' ') -> str:
	return str(binascii.hexlify(bline, sep), "utf-8")

def hex_to_str(bline: str, sep: str = ' ') -> str:
	return str(binascii.unhexlify(bline.replace(sep, '')), "utf-8")