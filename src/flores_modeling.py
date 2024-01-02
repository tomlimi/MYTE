import transformers
from transformers import ByT5Tokenizer, T5ForConditionalGeneration, T5Config

import torch
from tqdm import tqdm
import pandas as pd
import math
import math
import gc
import argparse
from pynvml import *

from myt5_tokenizer import MyT5Tokenizer

nvmlInit()
ALL_AVAILABLE_LANGUAGES = ['af', 'am', 'ar', 'az', 'be', 'bg', 'bn', 'ca', 'ceb', 'co', 'cs', 'cy', 'da', 'de', 'el', 'en', 'eo', 'es',
                 'et', 'eu', 'fa', 'fi', 'fil', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'haw','hi', 'hmn', 'ht', 'hu', 'hy',
                 'id', 'ig', 'is', 'it', 'iw', 'ja', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv',
                 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'no', 'ny', 'pa', 'pl', 'ps', 'pt', 'ro', 'ru',
                 'sd', 'si', 'sk', 'sl', 'sm', 'sn', 'so', 'sq', 'sr', 'st', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'uk', 'ur',
                 'uz', 'vi', 'xh', 'yi', 'yo', 'zh', 'zu']

FLORES_MAPPING = {'en': 'eng_Latn', 'ceb': 'ceb_Latn', 'de': 'deu_Latn', 'sv': 'swe_Latn', 'fr': 'fra_Latn', 'nl': 'nld_Latn', 'ru': 'rus_Cyrl', 'es': 'spa_Latn',
                    'it': 'ita_Latn', 'pl': 'pol_Latn', 'ja': 'jpn_Jpan', 'zh': 'zho_Hans', 'uk': 'ukr_Cyrl', 'vi': 'vie_Latn', 'ar': 'arb_Arab',
                    'pt': 'por_Latn', 'fa': 'pes_Arab', 'ca': 'cat_Latn', 'sr': 'srp_Cyrl', 'id': 'ind_Latn', 'ko': 'kor_Hang', 'no': 'nob_Latn',
                    'fi': 'fin_Latn', 'tr': 'tur_Latn', 'cs': 'ces_Latn', 'hu': 'hun_Latn', 'ro': 'ron_Latn', 'eu': 'eus_Latn', 'ms': 'zsm_Latn',
                    'eo': 'epo_Latn', 'he': 'heb_Hebr', 'hy': 'hye_Armn', 'da': 'dan_Latn', 'bg': 'bul_Cyrl', 'cy': 'cym_Latn', 'sk': 'slk_Latn',
                    'uz': 'uzn_Latn', 'et': 'est_Latn', 'be': 'bel_Cyrl', 'kk': 'kaz_Cyrl', 'el': 'ell_Grek', 'lt': 'lit_Latn', 'gl': 'glg_Latn',
                    'ur': 'urd_Arab', 'az': 'azj_Latn', 'sl': 'slv_Latn', 'ka': 'kat_Geor', 'hi': 'hin_Deva', 'th': 'tha_Thai', 'ta': 'tam_Taml',
                    'bn': 'ben_Beng', 'mk': 'mkd_Cyrl',  'lv': 'lvs_Latn', 'af': 'afr_Latn', 'tg': 'tgk_Cyrl', 'my': 'mya_Mymr',
                    'mg': 'plt_Latn', 'sq': 'als_Latn', 'mr': 'mar_Deva', 'te': 'tel_Telu', 'ml': 'mal_Mlym', 'ky': 'kir_Cyrl', 'sw': 'swh_Latn',
                    'jv': 'jav_Latn', 'ht': 'hat_Latn', 'lb': 'ltz_Latn', 'su': 'sun_Latn', 'ku': 'kmr_Latn', 'ga': 'gle_Latn', 'is': 'isl_Latn',
                    'fy': 'fao_Latn', 'pa': 'pan_Guru', 'yo': 'yor_Latn', 'ne': 'npi_Deva', 'ha': 'hau_Latn', 'kn': 'kan_Knda', 'gu': 'guj_Gujr',
                    'mn': 'khk_Cyrl', 'ig': 'ibo_Latn', 'si': 'sin_Sinh', 'ps': 'pbt_Arab', 'gd': 'gla_Latn', 'sd': 'snd_Arab', 'yi': 'ydd_Hebr',
                    'am': 'amh_Ethi', 'sn': 'sna_Latn', 'zu': 'zul_Latn', 'km': 'khm_Khmr', 'so': 'som_Latn', 'mi': 'mri_Latn',
                    'mt': 'mlt_Latn', 'lo': 'lao_Laoo', 'xh': 'xho_Latn', 'sm': 'smo_Latn', 'ny': 'nya_Latn', 'st': 'sot_Latn'}


def get_model_tokenizer(model_type, model_size, model_steps, checkpoint_dir, device=torch.device("cuda:0")):
	model = T5ForConditionalGeneration.from_pretrained(f"{checkpoint_dir}/{model_type}_{model_size}_{model_steps}", use_safetensors=True)
	model = model.to(device)
	if model_type == 'byt5':
		tokenizer = ByT5Tokenizer()
	else:
		tokenizer = MyT5Tokenizer(decompose_map="../byte_maps/decompose_map.json", merge_map="../byte_maps/merge_map.json")
	return model, tokenizer

def get_flores_data(flores_dir, languages, split='devtest'):
	flores = {}

	flores_split_dir = f'{flores_dir}/{split}'
	for lang in languages:
		with open(f'{flores_split_dir}/{FLORES_MAPPING[lang]}.{split}', 'r') as f:
			flores[lang] = f.read().splitlines()
	return flores

def print_gpu_mem_usage():
	h = nvmlDeviceGetHandleByIndex(0)
	info = nvmlDeviceGetMemoryInfo(h)
	print(f'total    : {info.total}')
	print(f'free     : {info.free}')
	print(f'used     : {info.used}')

def evaluate_texts(text_dataset, model, tokenizer, en_text_dataset=None, batch_size=32, context=0, translation=False, device=torch.device("cuda:0")):
	sentence_bpbs = []
	sentence_compressions = []
	context = min(abs(context), 1.0)

	for i in tqdm(range(0, len(text_dataset), batch_size)):
		batch = text_dataset[i:i + batch_size]
		batch_contexts = [math.floor(context * len(sent.split(" "))) for sent in batch]

		if translation and en_text_dataset is not None:
			context = 0
			batch_prefixes = en_text_dataset[i:i + batch_size]
		else:
			batch_prefixes = [" ".join(sent.split(" ")[:bc]) + " " for sent, bc in zip(batch, batch_contexts)]
		batch_suffixes = ["" + " ".join(sent.split(" ")[bc:]) for sent, bc in zip(batch, batch_contexts)]
		if en_text_dataset is not None:
			en_byte_lengths = torch.tensor(
				[len(sent.encode("utf-8")) + 1 for sent in en_text_dataset[i:i + batch_size]]).to(device)
		else:
			en_byte_lengths = None
		byte_lengths = torch.tensor([len(suf.encode("utf-8")) + 1 for suf in batch_suffixes]).to(device)

		if len(batch_prefixes) == 0:
			continue

		inputs = tokenizer(
			batch_prefixes, padding="longest", return_tensors="pt"
		).to(device)
		targets = tokenizer(
			batch_suffixes, padding="longest", return_tensors="pt"
		).to(device)

		outputs = model(**inputs, labels=targets.input_ids)

		logits = outputs.logits
		logits = torch.nn.functional.log_softmax(logits, dim=-1)

		target_labels = targets.input_ids.unsqueeze(-1)
		mask = targets.attention_mask

		target_logits = torch.gather(logits, -1, target_labels).squeeze(-1)

		if en_byte_lengths is not None:
			batch_bpbs = -torch.sum(mask * target_logits, axis=-1) / en_byte_lengths
		else:
			batch_bpbs = -torch.sum(mask * target_logits, axis=-1) / byte_lengths
		batch_compressions = torch.sum(mask, axis=-1) / byte_lengths
		# print(outputs.loss * batch_compressions)
		sentence_bpbs.extend(batch_bpbs.tolist())
		sentence_compressions.extend(batch_compressions.tolist())

		del batch_bpbs
		del byte_lengths
		del batch_compressions
		del targets
		del logits
		del mask
		del target_logits
		del inputs

	gc.collect()
	torch.cuda.empty_cache()
	return sentence_bpbs, sentence_compressions

def create_dfs(res_dict,model_name='', value_column='NLL'):
	data_list = []
	for lang, lang_vals in res_dict.items():
		for val in lang_vals:
			data_list.append([lang, model_name, val])
	data_df = pd.DataFrame(data_list, columns=['Language', 'Model', value_column])
	avg_df = data_df.groupby(['Language', 'Model'])[value_column].mean().reset_index()
	return data_df, avg_df

if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument("--languages", nargs="+", required=False, default=ALL_AVAILABLE_LANGUAGES)
	argparser.add_argument("--checkpoint_dir", required=False, default="../../hf_checkpoints", type=str)
	argparser.add_argument("--flores_dir", required=False, default="../flores200_dataset", type=str)
	argparser.add_argument("--results_dir", required=False, default="../flores200_lm_results", type=str)
	argparser.add_argument("--model_type", required=True, type=str)
	argparser.add_argument("--model_size", required=True, type=str)
	argparser.add_argument("--model_steps", required=False, type=int, default=250000)
	argparser.add_argument("--en_translation", action="store_true", default=False)

	args = argparser.parse_args()

	# load model + tokenizer
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model, tokenizer = get_model_tokenizer(args.model_type,args.model_size, args.model_steps, args.checkpoint_dir, device=device)
	flores = get_flores_data(args.flores_dir, args.languages, split='devtest')

	bpbs = dict()
	comps = dict()

	bs = 4 if args.model_size != 'large' else 2
	for lang in args.languages:
		print(f"Processing {lang}")
		bpbs[lang], comps[lang] = evaluate_texts(flores[lang], model, tokenizer, flores['en'], batch_size=bs,
		                                         context=0, translation=args.en_translation, device=device)
		print(f"BPB: {sum(bpbs[lang]) / len(bpbs[lang])}")
		print(f"Compression: {sum(comps[lang]) / len(comps[lang])}")
		print_gpu_mem_usage()

	# save results(nlls
	bpbs_data, bpbs_avg = create_dfs(bpbs,f"{args.model_type}_{args.model_size}", "BPEB")
	comp_data, comp_avg = create_dfs(comps,f"{args.model_type}_{args.model_size}", "Compressions")

	experiment_name = ""
	if args.model_steps != 250000:
		experiment_name += f"_{args.model_steps}"
	if args.en_translation:
		experiment_name += "_translate"

	bpbs_data.to_csv(f"{args.results_dir}/{args.model_type}_{args.model_size}{experiment_name}_bpeb.csv", index=False)
	bpbs_avg.to_csv(f"{args.results_dir}/{args.model_type}_{args.model_size}{experiment_name}_bpeb_avg.csv", index=False)

	comp_data.to_csv(f"{args.results_dir}/{args.model_type}_{args.model_size}_comp.csv", index=False)
	comp_avg.to_csv(f"{args.results_dir}/{args.model_type}_{args.model_size}_comp_avg.csv", index=False)



