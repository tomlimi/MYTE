import os
import json
import torch
import argparse
import time
import random
from tqdm import tqdm
import gc

from utils import normalize_text
from utils_modeling import get_model_tokenizer, print_gpu_mem_usage, create_dfs


TASK_LANGUAGES = {
	'translation': ['en2ta', 'en2te', 'en2el', 'en2hy', 'en2ru', 'en2kk', 'en2am', 'en2vi', 'en2ja', 'en2fr',
	                'en2ko', 'en2de', 'en2pl', 'en2sn'],
	'qa_in_lang': ['ar', 'bn', 'en', 'fi', 'id', 'ko', 'ru', 'sw', 'te'],
	'ner': ['am', 'bbj', 'bm', 'ee', 'ha', 'ig', 'lg', 'luo', 'mos', 'ny', 'pcm', 'rw', 'sn','sw','tn', 'tw', 'wo', 'xh', 'yo', 'zu'],
	'semantic_parsing': ['am', 'be', 'bn', 'de', 'en', 'es', 'fi', 'fr', 'ha', 'hi', 'ja',
	                     'pt_br', 'ru', 'sw', 'ta', 'th', 'tr', 'yo', 'zu'],
}


def parse_data_example(example, task):
	text, target = '', ''
	if task == 'ner':
		text = example['text']
		target = example['target']
	if task == 'qa_in_lang':
		text = ' '.join([example['context'], example['question']])
		target = example['target']
	if task == 'translation':
		text = example['input']
		target = example['target']
	if task == 'semantic_parsing':
		text = example['input']
		target = example['target']

	return {"text": normalize_text(text), "target": normalize_text(target)}


def get_dataset(lang, task, dataset_directory, sample_size=100, split='test'):

	data_file = os.path.join(dataset_directory, task, split, f"{lang}.jsonl")
	with open(data_file, 'r') as f:
		examples = [parse_data_example(json.loads(line), task) for line in f.readlines()]
	# shuffle and select a sample
	random.shuffle(examples)
	examples = examples[:sample_size]
	return examples


def stats_for_task(model, tokenizer, lang, task, dataset_directory, batch_size, device):

	examples = get_dataset(lang, task, dataset_directory)
	compressions = []
	times = []

	for i in tqdm(range(0, len(examples), batch_size)):
		batch = examples[i:i + batch_size]
		batch_texts = [ex['text'] for ex in batch]
		batch_targets = [ex['target'] for ex in batch]

		byte_lengths = torch.tensor([len(txt.encode("utf-8")) + len(tgt.encode("utf-8")) for txt, tgt in zip(batch_texts, batch_targets)]).to(device)

		inputs = tokenizer(
			batch_texts, padding="longest", return_tensors="pt", max_length=1024
		).to(device)
		targets = tokenizer(
			batch_targets, padding="longest", return_tensors="pt", max_length=1024
		).to(device)


		if device.type == "cuda":
			torch.cuda.synchronize()
		start = time.time()

		# _ = model(**inputs, labels=targets.input_ids)

		if device.type == "cuda":
			torch.cuda.synchronize()
		end = time.time()

		batch_compressions = (torch.sum(inputs.attention_mask, axis=-1) + torch.sum(targets.attention_mask, axis=-1))/ byte_lengths
		batch_times = [(end - start) / len(batch)] * len(batch)

		compressions.extend(batch_compressions.tolist())
		times.extend(batch_times)

		del byte_lengths
		del batch_compressions
		del targets
		del inputs

		gc.collect()
		torch.cuda.empty_cache()

	return compressions, times


if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument("--checkpoint_dir", required=False, default="../hf_checkpoints", type=str)
	argparser.add_argument("--dataset_directory", required=False, default="../xtreme_up_v1.1_lower", type=str)
	argparser.add_argument("--results_dir", required=False, default="../xtreme_up_results", type=str)

	argparser.add_argument("--task", required=True, type=str)
	argparser.add_argument("--model_type", required=True, type=str)
	argparser.add_argument("--model_size", required=True, type=str)
	argparser.add_argument("--model_steps", required=False, type=int, default=250000)

	args = argparser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model, tokenizer = get_model_tokenizer(args.model_type, args.model_size, args.model_steps, args.checkpoint_dir, device=device)

	comps = dict()
	times = dict()

	bs = 32
	for lang in TASK_LANGUAGES[args.task]:

		print(f"Processing {args.task} inference in {lang}")
		comps[lang], times[lang] = stats_for_task(model, tokenizer, lang, args.task, args.dataset_directory, batch_size=bs, device=device)
		print(f"Compression: {sum(comps[lang]) / len(comps[lang])}")
		print(f"Time: {sum(times[lang]) / len(times[lang])}")

	# save results(nlls
	comp_data, comp_avg = create_dfs(comps,f"{args.model_type}_{args.model_size}", "Compressions")
	times_data, times_avg = create_dfs(times,f"{args.model_type}_{args.model_size}", "Time")

	comp_data.to_csv(f"{args.results_dir}/{args.model_type}_{args.model_size}_{args.task}_comp.csv", index=True)
	comp_avg.to_csv(f"{args.results_dir}/{args.model_type}_{args.model_size}_{args.task}_comp_avg.csv", index=True)

	times_data.to_csv(f"{args.results_dir}/{args.model_type}_{args.model_size}_{args.task}_times.csv", index=True)
	times_avg.to_csv(f"{args.results_dir}/{args.model_type}_{args.model_size}_{args.task}_times_avg.csv", index=True)