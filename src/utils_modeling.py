import transformers
from transformers import ByT5Tokenizer, T5ForConditionalGeneration, T5Config
from pynvml import *
import torch
import pandas as pd

from myt5.myt5_tokenizer import MyT5Tokenizer


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
                  'mt': 'mlt_Latn', 'lo': 'lao_Laoo', 'xh': 'xho_Latn', 'sm': 'smo_Latn', 'ny': 'nya_Latn', 'st': 'sot_Latn',
                  'ast': 'ast_Latn', 'war': 'war_Latn', 'aeb': 'aeb_Arab', 'sa': 'san_Deva', 'sal': 'sat_Olck'}


LOW_RES_LANGUAGES = frozenset([ 'af', 'bn', 'be', 'bg', 'bs', 'my', 'ceb', 'da', 'et', 'gl', 'ka', 'el', 'he', 'id', 'kk', 'lv',
                                'lt', 'ms', 'pcm', 'ro', 'sk', 'sl', 'tl', 'ta', 'th', 'ug', 'uk', 'ur', 'uz', 'no', 'or', 'bho',
                                'brx', 'gbm', 'gom', 'hne', 'mai', 'mni', 'mwr', 'ps', 'ta', 'ff', 'rw', 'mr', 'mt', 'fil', 'ku',
                                'nb', 'oci', 'rup', 'sk', 'eu', 'sme', 'am', 'hy', 'as', 'ast', 'az', 'bm', 'bem', 'ber', 'my',
                                'ckb', 'ee', 'fon', 'ful', 'bbj', 'gu', 'ha', 'is', 'ig', 'ga', 'jv', 'kea', 'kam', 'kn', 'km',
                                'nw', 'ky', 'lo', 'ln', 'lij', 'olo', 'lg', 'luo', 'lb', 'mk', 'ml', 'mg', 'mi', 'mn', 'mos', 'nd',
                                'ne', 'nso', 'se', 'ny', 'oc', 'om', 'ps', 'pa', 'sa', 'gd', 'tn', 'sn', 'si', 'sd', 'so', 'ckb',
                                'st', 'ss', 'sw', 'tg', 'te', 'bo', 'ts', 'tw', 'umb', 'hsb', 've', 'cy', 'wo', 'xh', 'yo', 'zu',
                                'ast', 'war', 'aeb', 'sa', 'sal'])

def get_model_tokenizer(model_type, model_size, model_steps, checkpoint_dir, task=None, device=torch.device("cuda:0"), dropout=0.):
	# load fine-tuned model if available

	if task is not None and os.path.isdir(f"{checkpoint_dir}/{model_type}_{model_size}_{model_steps}_{task}") is True:
		model = T5ForConditionalGeneration.from_pretrained(f"{checkpoint_dir}/{model_type}_{model_size}_{model_steps}_{task}")
	else:
		model = T5ForConditionalGeneration.from_pretrained(f"{checkpoint_dir}/{model_type}_{model_size}_{model_steps}",
		                                                   use_safetensors=True, dropout_rate=dropout)
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


def create_dfs(res_dict, model_name='', value_column='NLL'):
	data_list = []
	for lang, lang_vals in res_dict.items():
		for val in lang_vals:
			data_list.append([lang, val])
	data_df = pd.DataFrame(data_list, columns=['Language', value_column])
	avg_df = data_df.groupby(['Language'])[value_column].mean().reset_index()

	# raname Language if its en2xx into xx (only in translation task)
	avg_df['Language'] = avg_df['Language'].apply(lambda x: x.split('2')[-1])

	avg_df.set_index('Language', inplace=True)
	avg_df.loc['AVG'] = avg_df.mean()
	avg_df.loc['AVG LR'] = avg_df[avg_df.index.isin(LOW_RES_LANGUAGES)].mean()
	return data_df, avg_df


def print_gpu_mem_usage():
	h = nvmlDeviceGetHandleByIndex(0)
	info = nvmlDeviceGetMemoryInfo(h)
	print(f'total    : {info.total}')
	print(f'free     : {info.free}')
	print(f'used     : {info.used}')
