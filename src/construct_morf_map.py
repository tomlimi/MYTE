from itertools import chain
import argparse
import json
import morfessor
from collections import Counter, OrderedDict, defaultdict

from typing import Iterable

# For the first code decompostion need
HEX_RANGES =[(0x0420, 0x05af),(0x0f50,0x0fff)]
LANG_CODE_LIMIT = 4096
TARGET_BYTE_PER_CODE = 3

def group_morfs(languages: Iterable[str], mtn: int, model_dir: str) -> list[dict]:
	pass

def merged_morf_map(languages: Iterable[str], mtn: int, model_dir: str) -> dict[str,str]:
	morf_mapping = {}
	morf_total_count = defaultdict(int)
	lang2hex_prefix = {}
	hex_codes_available = list(chain(*[range(*r) for r in HEX_RANGES]))

	for lang_id, lang in enumerate(languages):
		lang_hex_id= hex_codes_available[lang_id]
		lang2hex_prefix[lang] = lang_hex_id

		morf_lang_count = get_morf_count(lang, model_dir, mtn)
		# assert mtn <= 16**(TARGET_BYTE_PER_CODE*2 - 3), f"Too many codes for {lang}"
		print(f"Language {lang} has {len(morf_lang_count)} morfs appearing in total {sum(morf_lang_count.values())} times in lexicon")

		morf_dict = OrderedDict(morf_lang_count)
		morf_dict = {morf: count for morf, count in morf_lang_count.items() }	

		added_code_id = 0
		for morf, count in morf_dict.items():
			if len(morf) <= TARGET_BYTE_PER_CODE:
				continue
				# Adding morfs only longer than TARGET_BYTE_PER_CODE 
			morf_orth = " ".join(morf)
			if morf_orth in morf_mapping and morf_total_count[morf_orth] >= count:
				continue
				# TODO resolve colissions possibly by counts

			# rewrtie code is a 3 byte code used to replace the byte sequence.
			# TODO generalize to other number of bytes.
			assert TARGET_BYTE_PER_CODE == 3
			rewrite_code = f"{lang_hex_id:03x}{added_code_id:03x}"
			rewrite_code = rewrite_code[:2] + ' ' + rewrite_code[2:4] + ' ' + rewrite_code[4:]
			
			morf_mapping[morf_orth] = rewrite_code
			morf_total_count[morf_orth] = count
			added_code_id += 1

			if added_code_id >= 16**(TARGET_BYTE_PER_CODE*2 - 3):
				print(f"Warning: {lang} has more codes than suported, truncating")
				break

		print(f"Added {added_code_id} codes to the dictionary for {lang}")

	return morf_mapping


def get_morf_count(language: str, model_dir:str, mtn=4096):
	# load morfessor model
	model_file = f"{model_dir}/{language}_{mtn}.bin"
	# load morfessor model
	model = morfessor.MorfessorIO().read_binary_model_file(model_file)
	# get the set of tokens
	list_of_constructions = []
	for construction, _ in model._analyses.items():
		list_of_constructions.extend(model.segment(construction))
	
	return Counter(list_of_constructions)

if __name__ == "__main__":

	argparser = argparse.ArgumentParser()
	argparser.add_argument("--languages", nargs="+", required=True)
	argparser.add_argument("--mtn", type=int, required=False, default=4096)
	argparser.add_argument("--model_dir", required=True)
	argparser.add_argument("--mapping_dir", required=True)
	argparser.add_argument("--suffix", required=False, default="")
	args = argparser.parse_args()

	# TODO: group languages by morf similarrity
	# grouped_morfs = group_morfs(args.languages, args.mtn, args.model_dir)

	morf_mapping = merged_morf_map(args.languages, args.mtn, args.model_dir)

	# with open(f"{args.mapping_dir}/decompose_lc.json", "r") as f:
	# 	byte_map = json.load(f)

	# print("Byte map before update: ", len(byte_map))
	# byte_map.update(morf_mapping)
	# print("Byte map after update: ", len(byte_map))
	with open(f"{args.mapping_dir}/morf_map{args.suffix}.json", "w") as f:
		json.dump(morf_mapping, f, indent=4)
