# MYTE

## Modeling with MYTE: MyT5

For using tokenizer or and model install requirements from `requirements_myt5.txt`.


### Tokenizer

Custo implementation of MYTE tokenizer is in `src/myt5/myt5_tokenizer.py`. 

```
from src.myt5.myt5_tokenizer import MyT5Tokenizer

tokenizer = MyT5Tokenizer()
tokenized = tokenizer.tokenize("roughly at 12.")
# expected output: ['52', '82', 'a3', '93', '6c', '79', '20', '61', '74', '20', '31', '32', '2e']
```

You may also use custom byte maps for byte decomposition and merging (e.g. of morphemes) 
by providing custom merge and decompose maps.

```
from src.myt5.myt5_tokenizer import MyT5Tokenizer
tokenizer = MyT5Tokenizer(decompose_map="byte_maps/decompose_map.json", merge_map="byte_maps/merge_map.json")
```

### Model
Instruction for basic generation with myte model (MyT5).

```
from transformers import T5ForConditionalGeneration
from src.myt5.myt5_tokenizer import MyT5Tokenizer
import torch

my_t5 = T5ForConditionalGeneration.from_pretrained(PATH_TO_MODEL, use_safetensors=True)
tokenizer = MyT5Tokenizer()

pre_texts = ['"We now have',
            '„Mamy teraz myszy w wieku',
            '"""எங்களிடம் இப்போது']
post_texts = ['4-month-old mice that are non-diabetic that used to be diabetic," he added.',
              '4 miesięcy, które miały cukrzycę, ale zostały z niej wyleczone” – dodał.',
              '4-மாத-வயதுடைய எலி ஒன்று உள்ளது, முன்னர் அதற்கு நீரிழிவு இருந்தது தற்போது இல்லை"" என்று அவர் மேலும் கூறினார்."']

inputs = tokenizer(pre_texts, padding="longest", return_tensors="pt")
targets = tokenizer(post_texts, padding="longest", return_tensors="pt")


outputs = model(**inputs, labels=targets.input_ids)
probs = torch.nn.functional.oftmax(outputs.logits, dim=-1)

```

Note that the only difference betwean usage in MyT5 and ByT5 models is the tokenizer.
More documentation for ByT5 models on the HuggingFace model [page](https://huggingface.co/google/byt5-base).

## Citation

The code is associated with the [paper](https://arxiv.org/pdf/2403.10691.pdf):
```
@misc{limisiewicz2024myte,
      title={MYTE: Morphology-Driven Byte Encoding for Better and Fairer Multilingual Language Modeling}, 
      author={Tomasz Limisiewicz and Terra Blevins and Hila Gonen and Orevaoghene Ahia and Luke Zettlemoyer},
      year={2024},
      eprint={2403.10691},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```