from typing import List, Tuple, Any, Dict

import math
import random
import string

import datasets
import numpy as np
import sklearn
import sklearn.tree
import transformers
from collections import Counter
import torch
import time

def prepare_dataset(
  dataset: datasets.Dataset,
  text_column: str = "text",
  label_column: str = "label",
  template: str = 'Input: %s\nOutput: %s',
  num_shots: int = 128,
  num_prompts: int = 40,
  seed: int = 42,
) -> Tuple[List[str], Tuple[List[str], List[int]], List[str], str]:
  """Prepares a dataset for Tree Prompting.

  Args:
    dataset (datasets.Dataset) - the dataset to use for tree construction
    template (str) - template for combining data point and label
    column (str) - name of column to get text from, defaults to
      "text"
    label_column (str) - name of the column with label
    template (str) - template for combining data point and label
    num_shots (int) - number of data points in context per prompt
    num_prompts (int) - number of prompts to use for feature generation
    seed (int) - random seed

  Returns:
    prompts (List(str)) - the list of prompts for feature generation
    data (List(str), List(int)) - list of data points that will be used w/ prompts
      to create features
    labels (Dict[str, str]) - the class labels used in prompt generation,
      as a mapping from dataset labels to the string labels we assigned them
    template (str) - template to use for new data
  """    
  rng = np.random.default_rng(seed=seed)

  X_train_text = dataset[text_column]
  y_train = dataset[label_column]

  unique_ys = sorted(list(set(y_train)))
  examples_by_y = {}
  for y in unique_ys:
      examples_by_y[y] = sorted(
          list(filter(lambda ex: ex[1] == y, zip(X_train_text, y_train)))
      )
    
  num_labels = len(unique_ys)
  verbalizer = { y: L for L, y in zip(string.ascii_uppercase[:num_labels], unique_ys) }
  print("Verbalizer:", verbalizer)
  prompts = []

  # Create num_prompts prompts
  while len(prompts) < num_prompts:
    # Create a prompt with demonstration for each class
    prompt = ""
    chosen_examples = {
        y: rng.choice(
            examples,
            size=num_shots,
            replace=(len(examples) < num_shots),
        )
        for y, examples in examples_by_y.items()
    }

    # Take an even number of demonstrations per class, but shuffle them.
    demo_classes = unique_ys * \
        math.ceil(num_shots //
                  len(unique_ys))
    random.shuffle(demo_classes)
    demo_classes = demo_classes[:num_shots]

    for idx, y in enumerate(demo_classes):
        text, _ = chosen_examples[y][idx]
        prompt += template % (text, verbalizer[y]) + "\n"
    if prompt not in prompts:
        prompts.append(prompt)
  
  return prompts, (X_train_text, y_train), verbalizer, template


def create_features(prompts, data, labels, model,
                    tokenizer, verb_tokenized, template, batch_size = 20):
    data, labels = zip(*sorted(zip(data, labels), key=lambda x: len(x[0])))
    # Pretokenize and batch.
    data_new = []
    attn_old = []
    answers = []
    for i in range(0, (len(data)) - batch_size + 1, batch_size):
        prompts_batch = ["\n" + (template % (data[j], "")).strip()
                         for j in range(i, i + batch_size)]
        inputs = tokenizer(
            prompts_batch,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
            max_length=256,
            truncation=True
        )
        attn_old.append(inputs["attention_mask"].clone())
        data_new.append(inputs)
        answers.append(labels[i: i + batch_size])

    # Main loop
    prompt_kvs = []
    results = []
    for n, prompt in enumerate(prompts):
        print("Prompting: ", n)
        print(len(prompt))
        tokens = tokenizer(
            [prompt],
            return_tensors='pt',
            max_length=3000,
            truncation=True
        ).to("cuda")
        with torch.no_grad():
            outputs = model.forward(**tokens)
        kv = outputs["past_key_values"]
        prompt_kvs.append([(kv[0].to("cpu"), kv[1].to("cpu"))
                            for kv in outputs["past_key_values"]
                           ])
        # Setup prompt caching
        past_key_values_new = []
        past_key_values = outputs["past_key_values"]
        for i in range(len(past_key_values)):
            past_key_values_new.append(
                [
                    past_key_values[i][0].expand(batch_size, -1, -1, -1),
                    past_key_values[i][1].expand(batch_size, -1, -1, -1),
                ]
            )
        z = torch.zeros(
                batch_size, past_key_values[0][0].shape[-2]
        ).fill_(1).to("cuda")

        # Compute for each batch in dataset
        total_tokens = 0
        start = time.time()
        correct = 0
        total = 0
        local_result = []
        for i in range(len(data_new)):
            inputs = data_new[i].to("cuda")
            attention_mask = attn_old[i].to("cuda")
            pos = attention_mask.sum(-1)
            attention_mask = torch.cat((z, attention_mask,),dim=-1)
            inputs["attention_mask"] = attention_mask
            with torch.no_grad():
                q = model(**inputs, past_key_values=past_key_values_new)
                dist = q["logits"].softmax(-1)
            for k in range(len(prompts_batch)):
                token_output_position = pos[k].item() - 1
                total_tokens += token_output_position
                val = list([dist[k, token_output_position, v].item()
                            for _, v in verb_tokenized.items()])
                local_result.append(val)
                correct += torch.tensor(val).argmax().item() == answers[i][k]
                total += 1
            if i % 100 == 10-1:
                print("fs:", correct / total, "t/s: ", total_tokens / (time.time() - start))
        results.append(local_result)
    return results, prompt_kvs, labels

def load_awq_model():
  """
  requires `pip install autoawq` with GPU >= 8.0
  """
  from awq import AutoAWQForCausalLM
  from transformers import AutoTokenizer

  model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ"

  # Load model
  model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=False,
                                            trust_remote_code=False, safetensors=True)
  tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
  tokenizer.padding_side="right"
  tokenizer.pad_token = tokenizer.eos_token
  return model, tokenizer


def run_one(inputs, kv):
    inputs, attention_mask = inputs

    z = torch.zeros(
        1, kv[0][0].shape[-2]
    ).fill_(1).to("cuda")
    attention_mask = torch.cat((z, attention_mask,),dim=-1)
    inputs["attention_mask"] = attention_mask

    with torch.no_grad():
        q = model(**inputs, past_key_values=kv)
    return q["logits"][0, -1].softmax(-1)
    # return tokenizer.decode(q["logits"][0, -1].argmax())

cache = {}
CACHE = 3
feat_count = Counter()
def classify(example, tree, node_id, prompt_kvs, verb_tokenized):
    feat = tree.feature[node_id]
    L, R = tree.children_left[node_id], tree.children_right[node_id]
    is_split_node = L != R
    prompt = feat // len(verb_tokenized)
    check = feat % len(verb_tokenized)
    def get_kvs(prompt):
        feat_count[prompt] += 1
        if prompt in cache:
            return cache[prompt]
        else:
            if prompt in [a for a, _ in feat_count.most_common(CACHE)]:
                cache[prompt] = prompt_kvs[prompt].to("cuda")
                if len(cache) > CACHE:
                    del cache[feat_count.most_common(CACHE + 1)[-1][0]]
            return prompt_kvs[prompt].to("cuda")

    if is_split_node:
        dist = run_one(example, get_kvs(prompt))
        v = dist[verb_tokenized[check]].item()
        next = L if v <= tree.threshold[node_id] else R
        print(feat, prompt, check, v, tree.threshold[node_id])
        return classify(example, tree, next, prompt_kvs, verb_tokenized)
    else:
        print("BOTTOM", tree.value[node_id])
        ret = tree.value[node_id].argmax()
        return ret


DATAPOINTS = 2000
PROMPTS = 20
BATCH_SIZE = 2
dataset = datasets.load_dataset("rotten_tomatoes", "train")
dataset = dataset.shuffle(seed=42)
prompts, (data, labels), verb, template = prepare_dataset(dataset["train"].select(range(DATAPOINTS)), num_shots=128)

model, tokenizer = load_awq_model()
verb_tokenized = {}
for k,v in verb.items():
  verb_tokenized[k] = tokenizer.encode(v)[-1]

results, prompt_kvs, new_labels = create_features(prompts[:PROMPTS], data, labels, model, tokenizer,
                                                  verb_tokenized, template, batch_size=BATCH_SIZE)
print(verb_tokenized)
result = torch.tensor(results)
result = result.permute(1, 0, 2).contiguous().view(result.shape[1], -1)
print(result.shape)

clf = sklearn.tree.DecisionTreeClassifier(
    max_leaf_nodes = 40
)

clf.fit(result, new_labels)

prompt_kvs_stack = [torch.stack([torch.stack((kv[0], kv[1])) for kv in kv])
                    for kv in prompt_kvs]

correct = 0
total = 0
for i, p in enumerate(dataset["test"]):
    prompts_batch = ["\n" + (template % (p["text"], "")).strip()]
    inputs = tokenizer(
        prompts_batch,
        return_tensors="pt",
        return_attention_mask=True,
    ).to("cuda")

    correct += (classify((inputs, inputs["attention_mask"].clone()), clf.tree_, 0, prompt_kvs_stack, verb_tokenized) == p["label"])
    total += 1
    print(correct / total)
print("Final:", correct / total)
