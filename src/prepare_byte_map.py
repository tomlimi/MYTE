from itertools import chain
import argparse
import json

# For the first code decompostion need
HEX_RANGES =[(0x0420, 0x05af),(0x0f50,0x0fff)]
LANG_CODE_LIMIT = 4096
TARGET_BYTE_PER_CODE = 3

def load_token_set(lang,split, bp, ml, mtn):
	token_set = set()
	with open(f"../morfessor_out/{lang}/{split}_bp_{bp}_ml_{ml}_mtn_{mtn}.seg", "r") as segmented_file:
		for line in segmented_file:
			token_set.update(line.strip().split())
	return token_set


def merge_byte_maps(languages, split, bp, ml, mtn):
	merged_byte_map = {}
	lang2hex_prefix = {}
	hex_codes_available = list(chain(*[range(*r) for r in HEX_RANGES]))
	for lang_id, lang in enumerate(languages):
		lang_hex_id= hex_codes_available[lang_id]
		lang2hex_prefix[lang] = lang_hex_id

		code_set = load_token_set(lang, split, bp, ml, mtn)
		assert mtn <= 16**(TARGET_BYTE_PER_CODE*2 - 3), f"Too many codes for {lang}"
		print(f"Language {lang} has {len(code_set)} codes")

		added_code_id = 0
		for code in code_set:
			code = code.replace("-", " ")
			if (len(code) + 1) / 3 <= TARGET_BYTE_PER_CODE:
				continue
			if code in merged_byte_map:
				continue
			# rewrtie code is a 3 byte code used to replace the byte sequence.
			# TODO generalize to other number of bytes.
			assert TARGET_BYTE_PER_CODE == 3
			rewrite_code = f"{lang_hex_id:03X}{added_code_id:03X}".lower()
			rewrite_code = rewrite_code[:2] + ' ' + rewrite_code[2:4] + ' ' + rewrite_code[4:]
			merged_byte_map[code] = rewrite_code
			added_code_id += 1
			if added_code_id > 16**(TARGET_BYTE_PER_CODE*2 - 3):
				print(f"Warning: {lang} has more codes than suported, truncating")
				break

		print(f"Added {added_code_id} codes to the dictionary for {lang}")

	return merged_byte_map


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--byte_map_dir", default="../byte_maps")
	parser.add_argument("--languages", nargs="+", default=["eng_Latn", "spa_Latn", "por_Latn", "fra_Latn", "ita_Latn",
	                                                       "ron_Latn", "pol_Latn", "mlt_Latn", "heb_Hebr", "arb_Arab",
	                                                       "jpn_Jpan", "kor_Hang", "tel_Telu", "tam_Taml", "bod_Tibt",
	                                                       "sin_Sinh"])
	parser.add_argument("--split", default="dev")
	parser.add_argument("--bp", default=36)
	parser.add_argument("--ml", default=0)
	parser.add_argument("--mtn", default=4096)
	args = parser.parse_args()

	# open decompose json file for initial decomoposition of the codes
	with open(f"{args.byte_map_dir}/simple_decompose.json", "rb") as f:
		byte_map = json.load(f)

	merged_byte_map = merge_byte_maps(args.languages, args.split, args.bp, args.ml, args.mtn)

	# add the merged byte map to the initial decompose json file
	byte_map.update(merged_byte_map)

	with open(f"{args.byte_map_dir}/morfessor_bp_{args.bp}_ml_{args.ml}_mtn_{args.mtn}.json", "w") as f:
		json.dump(byte_map, f, indent=4)
