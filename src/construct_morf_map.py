from itertools import chain
import argparse
import json
import morfessor
from collections import OrderedDict, defaultdict
import numpy as np
from copy import copy

from typing import Iterable, Iterator
from utils import NUM_MORPH_CLUSTERS, get_morph_cluster

# For the first code decompostion need
HEX_RANGES = [(0x0420, 0x05af), (0x0f50, 0x0fff)]

B2_LEADING = 0x42
B3_LEADING = 0x4a
B4_LEADING = 0x52
LANG_CODE_LIMIT = 4096
TARGET_BYTE_PER_CODE = 3


def add_morph_mapping_from_iterator(morph_iterator: Iterator[tuple], cluster_id: int) -> dict[str, str]:
	cluster_morph_mapping = {}

	b1 = cluster_id + B2_LEADING
	for b2 in range(0x80, 0xbf):
		morph = next(morph_iterator)
		if morph is None:
			return cluster_morph_mapping
		if len(morph) <= 2:
			continue
		morph_orth = " ".join(morph)
		morph_mapping_code = f"{b1:02x} {b2:02x}"
		cluster_morph_mapping[morph_orth] = morph_mapping_code

	b1 = cluster_id + B3_LEADING
	for b2 in range(0x80, 0xbf):
		for b3 in range(0x80, 0xbf):
			morph = next(morph_iterator)
			if morph is None:
				return cluster_morph_mapping
			if len(morph) <= 3:
				continue
			morph_orth = " ".join(morph)
			morph_mapping_code = f"{b1:02x} {b2:02x} {b3:02x}"
			cluster_morph_mapping[morph_orth] = morph_mapping_code

	b1 = cluster_id + B4_LEADING
	for b2 in range(0x80, 0xbf):
		for b3 in range(0x80, 0xbf):
			for b4 in range(0x80, 0xbf):
				morph = next(morph_iterator)
				if morph is None:
					return cluster_morph_mapping
				if len(morph) <= 4:
					continue
				morph_orth = " ".join(morph)
				morph_mapping_code = f"{b1:02x} {b2:02x} {b3:02x} {b4:02x}"
				cluster_morph_mapping[morph_orth] = morph_mapping_code

	return cluster_morph_mapping


def get_clustered_mapping(clustered_morphs: dict[int, dict[tuple, float]]) -> dict[str, str]:
	morph_mapping = {}

	for cluster_id, cluster_morphs in clustered_morphs.items():
		# add morphs with highest scores as 3 byte mapping
		cluster_morphs = OrderedDict(sorted(cluster_morphs.items(), key=lambda x: x[1], reverse=True))
		morph_iterator = iter((*cluster_morphs.keys(), None))
		cluster_morph_mapping = add_morph_mapping_from_iterator(morph_iterator, cluster_id)
		print(f"For cluster: {cluster_id} added {len(cluster_morph_mapping)} morphs")
		morph_mapping.update(cluster_morph_mapping)
	return morph_mapping


def merged_morf_map(languages: Iterable[str], mtn: int, model_dir: str, sort_by_cost: bool, cluster_scripts: bool, method: str) -> dict[str,str]:
	morf_mapping = {}
	morf_total_score = defaultdict(int)
	clustered_morfs = {cluster_id: {} for cluster_id in range(NUM_MORPH_CLUSTERS)}

	lang2hex_prefix = {}
	hex_codes_available = list(chain(*[range(*r) for r in HEX_RANGES]))

	for lang_id, lang in enumerate(languages):
		lang_hex_id= hex_codes_available[lang_id]
		lang2hex_prefix[lang] = lang_hex_id

		if method == "morfessor":
			if sort_by_cost:
				morf_lang = get_morf_cost(lang, model_dir,True, mtn)
			else:
				morf_lang = get_morf_count(lang, model_dir,True, mtn)
		else:
			morf_lang = get_subwords(method, lang, model_dir, mtn)

		# assert mtn <= 16**(TARGET_BYTE_PER_CODE*2 - 3), f"Too many codes for {lang}"
		print(f"Language {lang} has {len(morf_lang)} morfs to add.")

		if not cluster_scripts:
			added_code_id = 0
			for morf, score in morf_lang.items():
				if len(morf) <= TARGET_BYTE_PER_CODE:
					continue
					# Adding morfs only longer than TARGET_BYTE_PER_CODE
				morf_orth = " ".join(morf)
				if morf_orth in morf_mapping and morf_total_score[morf_orth] >= score:
					continue
					# TODO resolve colissions possibly by scores

				# rewrtie code is a 3 byte code used to replace the byte sequence.
				# TODO generalize to other number of bytes.
				assert TARGET_BYTE_PER_CODE == 3
				rewrite_code = f"{lang_hex_id:03x}{added_code_id:03x}"
				rewrite_code = rewrite_code[:2] + ' ' + rewrite_code[2:4] + ' ' + rewrite_code[4:]

				morf_mapping[morf_orth] = rewrite_code
				morf_total_score[morf_orth] = score
				added_code_id += 1

				if added_code_id >= 16**(TARGET_BYTE_PER_CODE*2 - 3):
					print(f"Warning: {lang} has more codes than suported, truncating")
					break

			print(f"Added {added_code_id} codes to the dictionary for {lang}")
		else:
			for morph, score in morf_lang.items():
				if len(morph) <= TARGET_BYTE_PER_CODE:
					continue

				cluster_id = get_morph_cluster(morph)
				if cluster_id is None:
					continue
				if morph not in clustered_morfs[cluster_id] or clustered_morfs[cluster_id][morph] < score:
					clustered_morfs[cluster_id][morph] = score

	if cluster_scripts:
		morf_mapping = get_clustered_mapping(clustered_morfs)

	return morf_mapping


def get_morfessor_model(language: str, model_dir: str, mtn=4096):
	# load morfessor model
	model_file = f"{model_dir}/{language}_{mtn}.bin"
	# load morfessor model
	model = morfessor.MorfessorIO().read_binary_model_file(model_file)
	return model


def get_morf_count(language: str, model_dir:str, normalize:bool=True, mtn=4096):
	model = get_morfessor_model(language, model_dir, mtn)

	list_of_constructions = []
	construction_counts = []
	analyses = copy(model._analyses)

	for construction, node in analyses.items():
		count = node.count
		splitloc = node.splitloc
		# consider LEAF construction longer than 1 atom
		if not splitloc and len(construction) > 1:
			list_of_constructions.append(construction)
			construction_counts.append(float(count))

	construction_counts = np.array(construction_counts)
	if normalize:
		construction_counts /= construction_counts.sum()
	morf_count = {constr: count for constr, count in zip(list_of_constructions, construction_counts)}
	morf_count = OrderedDict(sorted(morf_count.items(), key=lambda x: x[1], reverse=True))

	return morf_count


def get_morf_cost(language: str, model_dir: str ,normalize: bool=True, mtn=4096):
	model = get_morfessor_model(language, model_dir, mtn)

	list_of_constructions = []
	construction_costs = []
	analyses = copy(model._analyses)
	base_cost = model.get_cost()

	for construction, node in analyses.items():
		count = node.count
		splitloc = node.splitloc
		# consider LEAF construction longer than 1 atom
		if not splitloc and len(construction) > 1:
			model._modify_construction_count(construction, -count)
			constuction_cost =  base_cost - model.get_cost()
			model._modify_construction_count(construction, count)
			list_of_constructions.append(construction)
			construction_costs.append(float(constuction_cost))

	construction_costs = np.array(construction_costs)
	if normalize:
		construction_costs /= construction_costs.sum()

	morf_cost = {constr: cost for constr, cost in zip(list_of_constructions, construction_costs)}
	morf_cost = OrderedDict(sorted(morf_cost.items(), key=lambda x: x[1], reverse=True))

	return morf_cost


def get_subwords(method, language, model_dir, mtn):
	tokenizer = f"{model_dir}/{method}_{language}_{mtn}.json"
	if method == "bpe":
		raise NotImplementedError

	# loading vocabulary
	tokenizer_dict = json.load(open(tokenizer, "r"))
	vocabulary = tokenizer_dict["model"]["vocab"][1:]
	vocabulary_costs = np.log(np.array([v[1] for v in vocabulary]))
	vocabulary_costs /= vocabulary_costs.sum()

	sw_cost = {sw_logit[0]: cost for sw_logit, cost in zip(vocabulary, vocabulary_costs)}
	sw_cost = OrderedDict(sorted(sw_cost.items(), key=lambda x: x[1], reverse=True))

	return sw_cost


if __name__ == "__main__":

	argparser = argparse.ArgumentParser()
	argparser.add_argument("--languages", nargs="+", required=True)
	argparser.add_argument("--mtn", type=int, required=False, default=4096)
	argparser.add_argument("--model_dir", required=True)
	argparser.add_argument("--mapping_dir", required=True)
	argparser.add_argument("--suffix", required=False, default="")
	argparser.add_argument("--sort_by_cost", action="store_true", default=False)
	argparser.add_argument("--cluster_scripts", action="store_true", default=False)
	argparser.add_argument("--method", type=str, default="morfessor")
	args = argparser.parse_args()

	# grouped_morfs = group_morfs(args.languages, args.mtn, args.model_dir)

	morf_mapping = merged_morf_map(args.languages, args.mtn, args.model_dir, args.sort_by_cost, args.cluster_scripts, args.method)

	with open(f"{args.mapping_dir}/morf_map{args.suffix}.json", "w") as f:
		json.dump(morf_mapping, f, indent=4)
