import torch
from tqdm import tqdm
import pandas as pd
import math
import math
import gc
import argparse
from pynvml import *
import time

from utils_modeling import get_model_tokenizer, get_flores_data, print_gpu_mem_usage, create_dfs

nvmlInit()
ALL_AVAILABLE_LANGUAGES = ['en', 'ceb', 'de', 'sv', 'fr', 'nl', 'ru', 'es', 'it', 'pl', 'ja', 'zh', 'uk', 'vi', 'ar', 'pt', 'fa', 'ca', 'sr',
            'id', 'ko', 'no', 'fi', 'tr', 'cs', 'hu', 'ro', 'eu', 'ms', 'eo', 'he', 'hy', 'da', 'bg', 'cy', 'sk', 'uz', 'et',
            'be', 'kk', 'el', 'lt', 'gl', 'ur', 'az', 'sl', 'ka', 'hi', 'th', 'ta', 'bn', 'mk', 'lv', 'af', 'tg', 'my',
            'mg', 'sq', 'mr', 'te', 'ml', 'ky', 'sw', 'jv', 'ht', 'lb', 'su', 'ku', 'ga', 'is', 'fy', 'pa', 'yo', 'ne', 'ha',
            'kn', 'gu', 'mn', 'ig', 'si', 'ps', 'gd', 'sd', 'yi', 'am', 'sn', 'zu', 'km', 'so', 'mi', 'mt', 'lo',
            'xh', 'sm', 'ny', 'st']



def evaluate_texts(text_dataset, model, tokenizer, en_text_dataset=None, batch_size=32, context=0, translation=False, device=torch.device("cuda:0")):
	sentence_bpbs = []
	sentence_compressions = []
	sentence_inference_times = []
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

		# compute runtime
		if device.type == "cuda":
			torch.cuda.synchronize()
		start = time.time()

		outputs = model(**inputs, labels=targets.input_ids)

		if device.type == "cuda":
			torch.cuda.synchronize()
		end = time.time()

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
		sentence_inference_times.extend([(end - start) / len(batch)] * len(batch))

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
	return sentence_bpbs, sentence_compressions, sentence_inference_times


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
	times = dict()

	bs = 4 if args.model_size != 'large' else 2
	for lang in args.languages:
		print(f"Processing {lang}")
		bpbs[lang], comps[lang], times[lang] = evaluate_texts(flores[lang], model, tokenizer, flores['en'], batch_size=bs,
		                                         context=0, translation=args.en_translation, device=device)
		print(f"BPB: {sum(bpbs[lang]) / len(bpbs[lang])}")
		print(f"Compression: {sum(comps[lang]) / len(comps[lang])}")
		print(f"Time: {sum(times[lang]) / len(times[lang])}")
		print_gpu_mem_usage()

	# save results(nlls
	bpbs_data, bpbs_avg = create_dfs(bpbs,f"{args.model_type}_{args.model_size}", "BPEB")
	comp_data, comp_avg = create_dfs(comps,f"{args.model_type}_{args.model_size}", "Compressions")
	times_data, times_avg = create_dfs(times,f"{args.model_type}_{args.model_size}", "Time")

	experiment_name = ""
	if args.model_steps != 250000:
		experiment_name += f"_{args.model_steps}"
	if args.en_translation:
		experiment_name += "_translate"

	bpbs_data.to_csv(f"{args.results_dir}/{args.model_type}_{args.model_size}{experiment_name}_bpeb.csv", index=False)
	bpbs_avg.to_csv(f"{args.results_dir}/{args.model_type}_{args.model_size}{experiment_name}_bpeb_avg.csv", index=False)

	comp_data.to_csv(f"{args.results_dir}/{args.model_type}_{args.model_size}_comp.csv", index=True)
	comp_avg.to_csv(f"{args.results_dir}/{args.model_type}_{args.model_size}_comp_avg.csv", index=True)

	times_data.to_csv(f"{args.results_dir}/{args.model_type}_{args.model_size}_times.csv", index=True)
	times_avg.to_csv(f"{args.results_dir}/{args.model_type}_{args.model_size}_times_avg.csv", index=True)



