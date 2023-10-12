import argparse
import codecs
import binascii


if __name__ == "__main__":

	argparse = argparse.ArgumentParser(description="Decode segmented corpus")
	argparse.add_argument("corpus_in", help="segmented corpus")
	argparse.add_argument("corpus_out", help="lexicon")
	argparse.add_argument("--escape_spaces", help="escape spaces", action="store_true")

	args = argparse.parse_args()

	with codecs.open(args.corpus_in, "r", "utf-8") as corpus_in:
		blines = corpus_in.readlines()

	# decode lines from the corpus
	lines = []
	n_segments = 0
	for bline in blines:
		line = []
		for segment in bline.split():
			line.append(binascii.unhexlify(segment.replace('-', '')).decode("utf-8", errors="backslashreplace"))
		if args.escape_spaces:
			line = [s for s in line if s != ' ']
		n_segments += len(line)
		lines.append(" ".join(line) + '\n')

	print(f"Avg. segments: {n_segments / len(blines):.2f}")

	with codecs.open(args.corpus_out, "w", "utf-8") as corpus_out:
		corpus_out.writelines(lines)
