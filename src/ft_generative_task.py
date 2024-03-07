from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
import accelerate
from accelerate import Accelerator
import argparse
import torch
import transformers
from functools import partial
import os
from copy import copy

from flores_modeling import get_model_tokenizer, get_flores_data
from utils import normalize_text
from tqdm import tqdm
import csv
from itertools import islice, cycle
from collections import defaultdict

N_EVAL_BATCHES = 50

TRANSLATION_LANGUAGES = ['ta', 'te', 'el', 'hy', 'ru', 'kk', 'am', 'vi', 'ja', 'fr', 'sm', 'st', 'ko', 'de', 'mt', 'pl', 'sn', 'en']


def preprocess_function(examples, tokenizer):
	padding = "max_length"
	max_length = 300

	inputs = [ex for ex in examples["Text"]]

	if "Expected" in examples:
		targets = [ex for ex in examples["Expected"]]
	else:
		targets = [""] * len(examples["Text"])
	model_inputs = tokenizer(inputs, max_length=max_length, padding=padding, truncation=True)
	labels = tokenizer(targets, max_length=max_length, padding=padding, truncation=True)

	model_inputs["labels"] = labels["input_ids"]
	return model_inputs


def reconstruct(inp, tokenizer, model):

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	result = []
	tokenized = tokenizer(inp, padding=True, return_tensors="pt").to(device)
	out = model.generate(**tokenized, max_length=300, early_stopping=True)
	out = out.cpu().numpy().tolist()
	for seq in out:
		seq = [i for i in seq if i != 0 and i != 1]
		result.append(normalize_text(tokenizer.decode(seq)))

	return result


def get_dataset(tokenizer, task, directory, shrads=10):

	if task == "spelling_correction":
		train_eval_dataset = load_dataset("csv", data_files=f"{directory}/train.csv")["train"]
		test_dataset = load_dataset("csv", data_files=f"{directory}/test.csv")["test"]
	elif task == "translation":
		flores_train_eval_data = get_flores_data(directory, TRANSLATION_LANGUAGES, split='dev')
		flores_test_data = get_flores_data(directory, TRANSLATION_LANGUAGES, split='devtest')

		train_eval_dict = defaultdict(list)
		test_dict = defaultdict(list)
		for lang in TRANSLATION_LANGUAGES:
			if lang == "en":
				continue
			train_eval_dict["Language"].extend([lang] * len(flores_train_eval_data[lang]))
			train_eval_dict["Text"].extend([f"<2{lang}> {en_sent}" for en_sent in flores_train_eval_data["en"]])
			train_eval_dict["Expected"].extend(flores_train_eval_data[lang])

			test_dict["Language"].extend([lang] * len(flores_test_data[lang]))
			test_dict["Text"].extend([f"<2{lang}> {en_sent}" for en_sent in flores_test_data["en"]])
			test_dict["Expected"].extend(flores_test_data[lang])

		train_eval_dataset = Dataset.from_dict(train_eval_dict)
		test_dataset = Dataset.from_dict(test_dict)
	else:
		raise ValueError("Invalid task")

	train_eval_dataset = train_eval_dataset.shuffle(seed=42)
	train_datasets = [train_eval_dataset.shard(num_shards=shrads, index=i) for i in range(shrads-1)]
	eval_dataset = train_eval_dataset.shard(num_shards=shrads, index=shrads-1)

	train_loaders = [DataLoader(dataset.map(partial(preprocess_function, tokenizer=tokenizer), batched=True, desc="Running tokenizer"), batch_size=4, shuffle=True) for dataset in train_datasets]
	eval_loader = DataLoader(eval_dataset.map(partial(preprocess_function, tokenizer=tokenizer), batched=True, desc="Running tokenizer"), batch_size=2, shuffle=True)
	test_loader = DataLoader(test_dataset.map(partial(preprocess_function, tokenizer=tokenizer), batched=True, desc="Running tokenizer"), batch_size=2, shuffle=True)

	return train_loaders, eval_loader, test_loader


def train_evaluate(model, train_loaders, eval_loader, lr=1e-4, n_epochs=30, orig_patience=2):
	patience=orig_patience
	accelerator = Accelerator(gradient_accumulation_steps=16, mixed_precision="bf16")
	device = accelerator.device
	model.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	optimizer = accelerate.optimizer.AcceleratedOptimizer(optimizer)
	scheduler = transformers.get_inverse_sqrt_schedule(optimizer, num_warmup_steps=100)
	train_loaders = list(islice(cycle(accelerator.prepare(train_loaders)), n_epochs))
	model, optimizer, scheduler, eval_loader = accelerator.prepare(model, optimizer, scheduler, eval_loader)
	for epoch, train_loader in enumerate(train_loaders):
		epoch_loss = 0.
		model.train()
		for batch in tqdm(train_loader):
			with accelerator.accumulate(model):
				optimizer.zero_grad()
				input_ids = torch.stack(batch['input_ids'], axis=1).to(device)
				attention_mask = torch.stack(batch['attention_mask'], axis=1).to(device)
				labels = torch.stack(batch['labels'], axis=1).to(device)
				outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
				loss = outputs.loss
				accelerator.backward(loss)
				optimizer.step()
				scheduler.step()
			epoch_loss += loss.item() / float(len(train_loader))
		# eval on a portion of the eval_loader
		del input_ids, attention_mask, labels, outputs, loss
		eval_loss = 0.
		model.eval()

		for batch in islice(eval_loader, N_EVAL_BATCHES):
			input_ids = torch.stack(batch['input_ids'], axis=1).to(device)
			attention_mask = torch.stack(batch['attention_mask'], axis=1).to(device)
			labels = torch.stack(batch['labels'], axis=1).to(device)
			outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
			eval_loss += outputs.loss.item() / float(N_EVAL_BATCHES)

		del input_ids, attention_mask, labels, outputs
		avg_train_loss = epoch_loss
		avg_eval_loss = eval_loss
		print(f'Epoch {epoch + 1}, Loss train: {avg_train_loss}, eval: {avg_eval_loss}')

		# Early stopping
		if epoch > 0 and avg_eval_loss > prev_loss:
			patience -= 1
		else:
			prev_loss = avg_eval_loss
			patience = orig_patience
			model_copy = copy(model)

		if patience == 0:
			print('Early stopping due to increase in loss')
			break

	return model_copy


def infer(model, tokenizer, evaluation_loader, model_type, model_size, directory,  split='test', experiment_name=None):

	if experiment_name is not None:
		out_file = f"{directory}/{split}_{model_type}_{model_size}_{experiment_name}.csv"
	else:
		out_file = f"{directory}/{split}_{model_type}_{model_size}.csv"
	with open(out_file, 'w', encoding='utf-8') as out:
		writer = csv.writer(out, quoting=csv.QUOTE_ALL)
		writer.writerow(["Id", "Language" ,"Predicted", "Expected"])
		for i, batch in tqdm(enumerate(evaluation_loader)):
			if "Id" not in batch:
				ids = [i * len(batch["Text"]) + i for j in range(len(batch["Text"]))]
			languages = batch["Language"]
			expected = batch["Expected"]
			reconstructed = reconstruct(batch["Text"], tokenizer, model)
			outputs = zip(ids, languages, reconstructed, expected)
			writer.writerows(outputs)


if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument("--checkpoint_dir", required=False, default="../../hf_checkpoints", type=str)

	argparser.add_argument("--task", required=True, type=str)
	argparser.add_argument("--directory", required=True, type=str)
	argparser.add_argument("--model_type", required=True, type=str)
	argparser.add_argument("--model_size", required=True, type=str)

	argparser.add_argument("--patience", default=2, type=int)
	argparser.add_argument("--lr", default=1e-4, type=float)
	argparser.add_argument("--experiment_name", default=None, type=str)
	argparser.add_argument("--model_steps", required=False, type=int, default=250000)

	args = argparser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	fine_tune = True
	if os.path.isdir(f"{args.checkpoint_dir}/{args.model_type}_{args.model_size}_{args.model_steps}_{args.task}") is True and args.experiment_name is None:
		print("Fine-tuned model exists, loading for evaluation")
		fine_tune = False
	model, tokenizer = get_model_tokenizer(args.model_type, args.model_size, args.model_steps, args.checkpoint_dir, device=device, task=args.task)

	train_loaders, eval_loader, test_loader = get_dataset(tokenizer, args.task, args.directory)

	if fine_tune:
		model = train_evaluate(model, train_loaders, eval_loader, lr=args.lr, orig_patience=args.patience)
		if args.experiment_name is None:
			model.save_pretrained(f"{args.checkpoint_dir}/{args.model_type}_{args.model_size}_{args.task}", use_safetensors=True)

	infer(model, tokenizer, eval_loader, args.model_type, args.model_size, args.directory,
	      split='dev', experiment_name=args.experiment_name)
	infer(model, tokenizer, test_loader, args.model_type, args.model_size, args.directory,
	      split='submission' if args.task == "spelling_correction" else 'devtest', experiment_name=args.experiment_name)