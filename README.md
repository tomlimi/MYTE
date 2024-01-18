## Code for Myte

Instruction for basic generation with myte model (MyT5).



```
from transformers import ByT5Tokenizer, T5ForConditionalGeneration
from src.myt5_tokenizer import MyT5Tokenizer
import torch

my_t5 = T5ForConditionalGeneration.from_pretrained(PATH_TO_MODEL, use_safetensors=True)
tokenizer = MyT5Tokenizer(decompose_map="byte_maps/decompose_map.json", merge_map="byte_maps/merge_map.json")

pre_texts = ['"We now have',
            '„Mamy teraz myszy w wieku',
            '"""எங்களிடம் இப்போது']
post_texts = ['4-month-old mice that are non-diabetic that used to be diabetic," he added.',
              '4 miesięcy, które miały cukrzycę, ale zostały z niej wyleczone” – dodał.',
              '4-மாத-வயதுடைய எலி ஒன்று உள்ளது, முன்னர் அதற்கு நீரிழிவு இருந்தது தற்போது இல்லை"" என்று அவர் மேலும் கூறினார்."']

inputs = tokenizer(pre_text, padding="longest", return_tensors="pt")
targets = tokenizer(pos_text, padding="longest", return_tensors="pt")


outputs = model(**inputs, labels=targets.input_ids)
probs = torch.nn.functional.oftmax(outputs.logits, dim=-1)

```

Note that the only difference betwean usage in MyT5 and ByT5 models is the tokenizer.
More documentation for ByT5 models on the HuggingFace model [page](https://huggingface.co/google/byt5-base).

