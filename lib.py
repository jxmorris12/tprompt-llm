from typing import List, Tuple

import math
import random
import string

import datasets
import numpy as np
import sklearn
import sklearn.tree
import ctranslate2
import transformers

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
) -> List[List[int]]:
  tokenizer = transformers.AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

  # Load LM
  generation_model_path = './models/ctranslate2'
  generator = ctranslate2.Generator(generation_model_path, device='cuda')

  data_train = data[0]
  features = [[None for _ in range(len(data_text))] for _ in range(len(prompts))]

  for i, system_prompt in enumerate(prompts):
    start_tokens = [tokenizer.convert_ids_to_tokens(tokenizer.encode(system_prompt + 'Input: ' + item + '\nOutput:')) for item in data_train]
    s_tokens = []
    for b in start_tokens:
      if len(b) > 2048:
        b = b[-2048:]
      s_tokens.append(b)
    start_tokens = s_tokens

    step_results = generator.generate_batch(
      #prompt=prompt_tokens,
      #static_prompt=system_prompt_tokens,
      start_tokens=start_tokens,
      max_length=2,
      sampling_topk=10,
      sampling_temperature=0.7
    )
    for j, step_result in enumerate(step_results):
      features[i][j] = step_result.sequences
  return features



class PromptTree:
  llm: Any # language model
  _clf: sklearn.tree.DecisionTreeClassifier # internal decision tree
  def __init__(
      self, 
      llm: Any,
      prompts: List[str],
      data: List[str],
      labels: List[str],
      verbalizer: Dict[str, str],
      max_leaf_nodes: int = 10,
      seed: int = 42
    ):
      self.llm = llm
      self.prompts = prompts
      prompt_cache, features = create_features(
          llm=llm, prompts=prompts, data=data, labels=labels
      )
      self._clf = sklearn.tree.DecisionTreeClassifier(
        max_leaf_nodes=max_leaf_nodes,
        random_state=seed,
      )
      X, y = data
      self._clf_.fit(X, y)
    
  def _features_from_text(self, text: List[str]) -> List[List[int]]:
    raise NotImplementedError

  def predict_proba(self, X_text):
      prompt_features = self._features_from_text(
          X_text
      )
      return self.clf_.predict_proba(prompt_features)

  def predict(self, X):
      labels = self.predict_proba(X).argmax(axis=1)
      return map(self.verbalizer.get, labels)


prompts, data, verbalizer = prepare_dataset(dataset["train"])
tree = PromptTree(
    llm=llm,
    prompts=prompts,
    data=data,
    verbalizer=verbalizer,
)
