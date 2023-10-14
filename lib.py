from typing import List, Tuple

import math
import random
import string

import datasets
import numpy as np
import sklearn
import sklearn.tree

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

def create_features(
  prompts: List[str],
  data: List[str],
  labels: List[str],
  model, tokenizer, batch_size = 20
) -> List[List[int]]:
    
    # Pretokenize and batch.
    data_new = []
    attn_old = []
    answers = []
    for i in range(0, (len(data)) - batch_size,batch_size):
        prompts_batch = ["Input: " + data[j] + "\nOutput:" for j in range(i, i + batch_size)]
        inputs = tokenizer(
            prompts_batch,
            return_tensors="pt",
            return_attention_mask=True,
            padding=True,
            max_length=128,
            truncation=True
        )
        attn_old.append(inputs["attention_mask"].clone())
        data_new.append(inputs)
        answers.append(labels[i: i + batch_size])

    # Main loop
    results = []
    for i, prompt in enumerate(prompts):
        print("Prompting: ", i)
        tokens = tokenizer(
            [prompt + ""],
            return_tensors='pt',
            max_length=2000,
            truncation=True
        ).to("cuda")
        with torch.no_grad():
            outputs = model.forward(**tokens)
        
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
        local_result = []
        for i in range(len(data_new)):
            inputs = data_new[i].to("cuda")        
            attention_mask = attn_old[i].to("cuda")
            pos = attention_mask.sum(-1)
            attention_mask = torch.cat((z, attention_mask,),dim=-1)
            inputs["attention_mask"] = attention_mask
            with torch.no_grad():
                q = model(**inputs, past_key_values=past_key_values_new)
            for k in range(len(prompts_batch)):
                token_output_position = pos[k].item() - 1
                total_tokens += token_output_position
                val = tokenizer.decode(q["logits"][k, token_output_position].argmax())
                local_result.append(val)
                correct += (val == ("B" if answers[i][k] else "A"))
            if i % 10 == 10-1:
                print(correct, "t/s: ", total_tokens / (time.time() - start))
        results.append(local_result)
    return results

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
  

def build_tree(
  features: List[List[int]]
) -> sklearn.tree.DecisionTreeClassifier:  
  raise NotImplementedException


prompts, data, labels = prepare_dataset(dataset["train"])
