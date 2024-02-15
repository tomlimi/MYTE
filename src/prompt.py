#!/usr/bin/env python
# -*- coding: utf-8 -*- 

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import T5ForConditionalGeneration, ByT5Tokenizer
# from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import torch
import torch.nn.functional as F
from datasets import load_dataset, DatasetDict

from myt5_tokenizer import MyT5Tokenizer

import argparse
from tqdm import tqdm
import os
import numpy as np
import random
import pickle
import math

LABELS = {
	'xnli': ['Yes', 'Also', 'No'],  # ['entailment', 'neutral', 'contradiction']
	'xstorycloze': None,
	'northeurlex': None
}

LANGS = {
	'xnli': ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh'],
	'xstorycloze': ['ar', 'en', 'es', 'eu', 'hi', 'id', 'my', 'ru', 'sw', 'te', 'zh'],
	'northeurlex': ['ru', 'fr', 'es', 'tr', 'ko', 'ja', 'ar', 'de']
}

NUM_RUNS = 5
NUM_TRANSLATION_OPTIONS = 4
NUM_EVAL_EXAMPLES = 1000  # fixed across runs
ISO_TO_NAME_MAP = {'ru': 'Russian', 'en': 'English', 'fr': 'French', 'es': 'Spanish', 'tr': 'Turkish', 'ko': 'Korean',
                   'ja': 'Japanese', 'ar': 'Arabic', 'de': 'German'}


def _create_context(task, k, split, eval_lang, datapath, rand_seed, run_id):
	context = ''
	if k == 0: return context

	data = load_dataset(task, eval_lang, data_dir=datapath)[split]
	data = data.shuffle(seed=rand_seed + run_id)
	data = data.select(list(range(k)))

	for example in data:
		if task == 'xnli':
			gold_label = LABELS['xnli'][int(example['label'])]
		elif task == 'xstorycloze':
			gold_label = example['sentence_quiz2'] if example['answer_right_ending'] == 2 else example['sentence_quiz1']
		elif task == 'northeurlex':
			gold_label = example['tgt']
		example_text = _create_example(example, task, gold_label, eval_lang)
		context += example_text[0] + example_text[1] + '\n'

	return context


def _create_example(example, task, label, eval_lang=None):
	# based on buffet prompt
	if task == 'xnli':
		# format from xnli eval of XGLM
		prompt_template_shared = "{}, right? "
		prompt_template_scored = "{}, {}"
		example_text0 = prompt_template_shared.format(example['premise'])
		example_text1 = prompt_template_scored.format(label, example['hypothesis'])

	elif task == 'xstorycloze':
		# format from xstorycloze eval of XGLM
		prompt_template_shared = "{} {} {} {} "
		prompt_template_scored = "{}"
		example_text0 = prompt_template_shared.format(example['input_sentence_1'], example['input_sentence_2'],
		                                              example['input_sentence_3'], example['input_sentence_4'])
		example_text1 = prompt_template_scored.format(label)

	elif task == 'northeurlex':
		# format from north_eurlex eval of mBERT (https://arxiv.org/pdf/2010.08275.pdf#page3)
		prompt_template_shared = "The word '{}' in {} is: "
		prompt_template_scored = "{}"
		example_text0 = prompt_template_shared.format(example['src'], ISO_TO_NAME_MAP[eval_lang])
		example_text1 = prompt_template_scored.format(label)

	return (example_text0, example_text1)


def score(model, tokenizer, input_text, context_ids=None):
	scores = []

	# process shared subset first and get model state
	shared_text = input_text[0][0]

	shared_ids = tokenizer(shared_text, padding="longest", return_tensors="pt")['input_ids'].to("cuda:0")
	if context_ids is not None:
		shared_ids = torch.cat([context_ids, shared_ids], dim=1)

	# score example with each possible label
	for _, input_choice in input_text:
		# tokenize input
		labels = tokenizer(input_choice, padding="longest", return_tensors="pt")['input_ids'].to("cuda:0")
		labels[labels == tokenizer.pad_token_id] = -100

		with torch.no_grad():
			loss = model(shared_ids, labels=labels).loss
		scores.append(loss.item())

	return scores


def _calculate_acc(scores):
	acc = [1 if int(y) == probs.index(min(probs)) else 0 for y, probs in scores]
	acc = (sum(acc) / len(acc)) * 100
	return acc


def get_model_tokenizer(model_type, model_size, model_steps, checkpoint_dir, device=torch.device("cuda:0")):
	model = T5ForConditionalGeneration.from_pretrained(f"{checkpoint_dir}/{model_type}_{model_size}_{model_steps}", use_safetensors=True)
	model = model.to(device)
	if model_type == 'byt5':
		tokenizer = ByT5Tokenizer()
	else:
		tokenizer = MyT5Tokenizer(decompose_map="../byte_maps/decompose_map.json", merge_map="../byte_maps/merge_map.json")
	return model, tokenizer


def main(args):
	eval_task = args.task
	datapath = None

	if eval_task == 'xnli':
		context_split = 'validation'
		eval_split = 'test'
	elif eval_task == 'xstorycloze':
		context_split = 'train'
		eval_split = 'validation'
		datapath = './xstorycloze'
	elif eval_task == 'northeurlex':
		context_split = 'train'
		eval_split = 'test'
		datapath = './northeurlex'

	# load model
	models, tokenizer = get_model_tokenizer(args.model_type, args.model_size, args.model_steps, args.checkpoint_dir, device=torch.device("cuda:0"))

	for eval_lang in args.eval_lang:

		# demonstration langs
		demo_lang = args.demo_lang if args.demo_lang != None else eval_lang

		# load dataset
		if eval_task == 'northeurlex':
			data = DatasetDict.load_from_disk("{}/{}".format(datapath, eval_lang))[eval_split]
		else:
			data = load_dataset(eval_task, eval_lang, data_dir=datapath)[eval_split]

		data = data.shuffle(seed=args.rand_seed)
		data = data.select(list(range(min(NUM_EVAL_EXAMPLES, len(data)))))

		task_acc_by_run = []

		# multiple runs!!!
		for run_id in range(NUM_RUNS):
			if args.k == 0 and run_id > 0:
				break

			# skip run if we already have results saved
			scores_filepath = '{}.{}.k{}.eval_{}.run{}.pkl'.format(args.model_type, eval_task, args.k, eval_lang, run_id)
			scores_filepath = os.path.join(args.output_dir, scores_filepath)
			if os.path.isfile(scores_filepath):
				print('Skipping run {} of {}...'.format(run_id, NUM_RUNS))

				with open(scores_filepath, 'rb') as f:
					scores = pickle.load(f)
				acc = _calculate_acc(scores)
				task_acc_by_run.append(acc)
				continue

			context_ids = None
			if args.k > 0:
				# create in-context examples text
				context = _create_context(eval_task, args.k, context_split, demo_lang, datapath, args.rand_seed, run_id)
				# cache context values to save on compute
				context_ids = tokenizer(context, padding="longest", return_tensors="pt")['input_ids'].to("cuda:0")

			# for each example -- get weight over all possible labels from all langs
			scores = []

			data_indices = list(range(len(data)))
			for idx, example in enumerate(tqdm(data)):
				# format example for input
				if eval_task in ['xnli']:
					example_texts = [_create_example(example, eval_task, label=label_option) for label_option in
					                 LABELS[eval_task]]
					gold_label_idx = int(example['label'])
				elif eval_task == 'xstorycloze':
					example_texts = [_create_example(example, eval_task, label=example['sentence_quiz1']),
					                 _create_example(example, eval_task, label=example['sentence_quiz2'])]
					gold_label_idx = example['answer_right_ending'] - 1  # convert to 0-indexed
				elif eval_task == 'northeurlex':
					example_texts = [_create_example(example, eval_task, label=example['tgt'], eval_lang=eval_lang)]
					# sample alternative examples
					sample_option_indices = random.sample(data_indices[:idx] + data_indices[idx + 1:],
					                                      NUM_TRANSLATION_OPTIONS - 1)
					example_texts.extend(
						[_create_example(example, eval_task, label=alt_exmpl['tgt'], eval_lang=eval_lang) for alt_exmpl
						 in data.select(sample_option_indices)])
					gold_label_idx = 0

				x = score(models, tokenizer, example_texts, context_ids)
				scores.append((gold_label_idx, x))

			# write out scores to file!
			print(scores_filepath)
			with open(scores_filepath, 'wb') as f:
				pickle.dump(scores, f)

			# calculate accuracy
			acc = _calculate_acc(scores)
			task_acc_by_run.append(acc)

		# get avg, var, std err across runs
		mean = sum(task_acc_by_run) / len(task_acc_by_run)
		var = sum([(s - mean) ** 2 for s in task_acc_by_run]) / len(task_acc_by_run)
		std_dev = math.sqrt(var)
		std_err = std_dev / math.sqrt(len(task_acc_by_run))
		print('{} {} Acc: {:3.2f} ± {:4.3f}'.format(eval_task, eval_lang, mean, std_err))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--rand_seed", default=42, type=int
	)
	parser.add_argument(
		"--k", default=0, type=int,
		choices=[0, 1, 4, 8]
	)
	parser.add_argument(
		"--eval_lang", required=True, type=str, nargs='+'
	)
	parser.add_argument(
		"--demo_lang", type=str,
	)
	parser.add_argument(
		"--task", required=True, type=str,
		choices=['xnli', 'xstorycloze', 'northeurlex']
	)
	parser.add_argument(
		"--output_dir", default='.', type=str,
	)
	parser.add_argument(
		"--checkpoint_dir", required=True, type=str,
	)
	parser.add_argument(
		"--model_type", required=True, type=str,
		choices=['myt5', 'byt5']
	)
	parser.add_argument(
		"--model_size", default='large', type=str,
		choices=['base', 'large']
	)
	parser.add_argument(
		"--model_steps", default=250000, type=int,
	)


	args = parser.parse_args()
	print(args)

	torch.manual_seed(args.rand_seed)
	os.environ['PYTHONHASHSEED'] = str(args.rand_seed)
	torch.cuda.manual_seed(args.rand_seed)
	torch.cuda.manual_seed_all(args.rand_seed)
	np.random.seed(args.rand_seed)
	random.seed(args.rand_seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

	main(args)