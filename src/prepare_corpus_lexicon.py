# prepare morfessor lexicon from the training corpus

import codecs
import argparse
import binascii
import itertools



if __name__ == "__main__":
	argparser = argparse.ArgumentParser(description="Prepare morfessor lexicon from the training corpus")
	argparser.add_argument("corpus_in", help="training corpus")
	argparser.add_argument("corpus_out", help="output corpus")
	argparser.add_argument("--lexicon", help="output lexicon", default=None)
	argparser.add_argument("--byte_patches", help="use patches instead of words", action="store_true")
	argparser.add_argument("--patch_size", help="patch size", type=int, default=24)

	args = argparser.parse_args()

	# reaa lines from the corpus
	with codecs.open(args.corpus_in, "r", "utf-8") as corpus_file:
		lines = corpus_file.readlines()

	# word counts and save as lexicon
	compound_counts = {}

	compounds = []
	blines = []
	for line in lines:
		bline=bytes(line.strip(), 'utf-8')
		line_compounds = bline.split()
		if args.byte_patches:
			line_compounds = list(itertools.chain.from_iterable([[comp[i:i+args.patch_size] for i in range(0, len(comp), args.patch_size)]
			                  for comp in line_compounds]))

		compounds.extend(line_compounds)
		blines.append(bline)

	for comp in compounds:
		if comp not in compound_counts:
			compound_counts[comp] = 0
		compound_counts[comp] += 1

	# sort by counts
	compound_counts = {k: v for k, v in sorted(compound_counts.items(), key=lambda item: item[1], reverse=True)}
	if args.lexicon is not None:
		with codecs.open(args.lexicon, "w") as lexicon_file:
			for comp, count in compound_counts.items():
				b_comp = str(binascii.hexlify(comp, '-'), "utf-8")
				lexicon_file.write(f"{count}\t{b_comp}\n")
		print(f"Lexicon saved to {args.lexicon}")

	# save corpus
	with codecs.open(args.corpus_out, "w", "utf-8") as corpus_file:
		for bline in blines:
			corpus_file.write(str(binascii.hexlify(bline, '-'), "utf-8") + "\n")
	print(f"Byte corpus saved to {args.corpus_out}")