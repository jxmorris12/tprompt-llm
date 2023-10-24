from dataclasses import dataclass
from sklearn.tree import DecisionTreeClassifier
import datasets
from typing import Any, Dict, Optional, List, Tuple
import math
import numpy as np
import random 
import string
import torch
import time 
from collections import Counter
import pickle

@dataclass
class TreeState:
    tree: Optional[Any]
    verbalizer: Optional[Dict[int, str]]
    template: str
    prompts: Optional[List[str]]
    llm: str
    prompt_max_length: int

class PromptTree:
    def __init__(self,   
                llm: str = "TheBloke/Mistral-7B-Instruct-v0.1-AWQ",
                template: str = 'Input: %s\nOutput: %s',
                num_shots: int = 128,
                num_prompts: int = 40,
                max_leaf_nodes: int = 40,
                prompt_max_length: int = 3000,
                cache_size: int = 3,
                seed: int = 42):
        """
        template (str) - template for combining data point and label
        num_shots (int) - number of data points in context per prompt
        num_prompts (int) - number of prompts to use for feature generation
        seed (int) - random seed
        """
        self.state = TreeState(None, None, template, None, llm, prompt_max_length)
        self.num_shots = num_shots
        self.num_prompts = num_prompts
        self.seed = seed

        self.cache = {}
        self.CACHE = cache_size
        self.feat_count = Counter()

        # These are caches
        self._prompt_kvs = None
        self._model = None
        self._tokenizer = None
        self._verb_tokenized = None

    def load(self, path: str):
        self.state = pickle.load(open(path, "rb"))

    def save(self, path : str):
        pickle.dump(self.state, open(path, "wb"))

    def _fill_template(self, x:str) -> str:
        return "\n" + (self.state.template % (x, "")).strip()

    def fit(self, 
            dataset: datasets.Dataset,
            text_column: str = "text",
            label_column: str = "label",
            batch_size: int = 20
            ):
        X_train_text, y_train = \
            self._gen_data(dataset[text_column], dataset[label_column])
        features = self._create_features(X_train_text, y_train, 
                                         batch_size)
        clf = DecisionTreeClassifier(max_leaf_nodes = 40)
        clf.fit(features, y_train)
        self.state.tree = clf.tree_

    def _gen_data(self, X: List[str], y: List[int]):
        rng = np.random.default_rng(seed=self.seed)

        unique_ys = sorted(list(set(y)))
        examples_by_y = {}
        for ys in unique_ys:
            examples_by_y[ys] = sorted(
                list(filter(lambda ex: ex[1] == ys, zip(X, y)))
            )
            
        num_labels = len(unique_ys)
        self.state.verbalizer = { ys: L for L, ys in zip(string.ascii_uppercase[:num_labels], unique_ys) }
        self.state.prompts = []

        # Create num_prompts prompts
        while len(self.state.prompts) < self.num_prompts:
            # Create a prompt with demonstration for each class
            prompt = ""
            chosen_examples = {
                ys: rng.choice(
                    examples,
                    size=self.num_shots,
                    replace=(len(examples) < self.num_shots),
                )
                for ys, examples in examples_by_y.items()
            }

            # Take an even number of demonstrations per class, but shuffle them.
            demo_classes = unique_ys * \
                math.ceil(self.num_shots //
                        len(unique_ys))
            random.shuffle(demo_classes)
            demo_classes = demo_classes[:self.num_shots]

            for idx, ys in enumerate(demo_classes):
                text, _ = chosen_examples[ys][idx]
                prompt += self.state.template % (text, self.state.verbalizer[ys]) + "\n"
            if prompt not in self.state.prompts:
                self.state.prompts.append(prompt)
    
        return zip(*sorted(zip(X, y), key=lambda x: len(x[0])))
        
    @property 
    def tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        self._model, self._tokenizer = self._load_llm()
        return self._tokenizer

    @property 
    def model(self):
        if self._model is not None:
            return self._model
        self._model, self._tokenizer = self._load_llm()
        return self._model

    @property
    def prompt_kvs(self):
        if self._prompt_kvs is not None:
            return self._prompt_kvs
        prompt_kvs = []
        for n, prompt in enumerate(self.state.prompts):
            print("Prompting: ", n)
            outputs = self._prompt_kv(n)
            kv = outputs["past_key_values"]
            prompt_kvs.append([(kv[0].to("cpu"), kv[1].to("cpu"))
                                for kv in outputs["past_key_values"]
                            ])
        self._prompt_kvs = [torch.stack([torch.stack((kv[0], kv[1])) for kv in kv])
                            for kv in prompt_kvs]
        return self._prompt_kvs

    @property
    def verb_tokenized(self):
        if self._verb_tokenized is not None:
            return self._verb_tokenized
        verb_tokenized = {}
        for k,v in self.state.verbalizer.items():
            verb_tokenized[k] = self.tokenizer.encode(v)[-1]
        self._verb_tokenized = verb_tokenized
        return self._verb_tokenized

    def _prompt_kv(self, i: int) -> torch.Tensor:
        "Create a kv tensor for prompt i"
        tokens = self.tokenizer(
                [self.state.prompts[i]],
                return_tensors='pt',
                max_length=self.state.prompt_max_length,
                truncation=True
            ).to("cuda")
        with torch.no_grad():
            outputs = self.model.forward(**tokens)
        return outputs
        
    def _create_features(self, data: List[str], labels: List[int], batch_size: int) -> torch.Tensor:
        model, tokenizer = self._load_llm()
        verb_tokenized = self.verb_tokenized

        # Pretokenize and batch.
        data_new = []
        attn_old = []
        answers = []
        for i in range(0, (len(data)) - batch_size + 1, batch_size):
            prompts_batch = [self._fill_template(data[j]) for j in range(i, i + batch_size)]
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
        for n, prompt in enumerate(self.state.prompts):
            print("Prompting: ", n)
            print(len(prompt))
            outputs = self._prompt_kv(n)

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
        results = torch.tensor(results)
        results = results.permute(1, 0, 2).contiguous().view(results.shape[1], -1)
        
        self._prompt_kvs = [torch.stack([torch.stack((kv[0], kv[1])) for kv in kv])
                            for kv in prompt_kvs]
        return results

    def predict(self,             
             dataset: datasets.Dataset,
             text_column: str = "text",
             label_column: str = "label"):
        correct = 0
        total = 0
        for i, p in enumerate(dataset):
            prompts_batch = [self._fill_template(p["text"])]
            inputs = self.tokenizer(
                prompts_batch,
                return_tensors="pt",
                return_attention_mask=True,
            ).to("cuda")
            correct += (self._classify((inputs, inputs["attention_mask"].clone()), 0) == p["label"])
            total += 1
            if i % 100 == 0:
                print(correct / total)
        print("Final:", correct / total)


    def _load_llm(self) -> Tuple[Any, Any]:
        if self._model is not None:
            return self._model, self._tokenizer
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
        model_name_or_path = self.state.llm

        # Load model
        model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=False,
                                                    trust_remote_code=False, safetensors=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
        tokenizer.padding_side="right"
        tokenizer.pad_token = tokenizer.eos_token
        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer

    def _run_one(self, inputs: Tuple[str, torch.Tensor], prompt: int) -> torch.Tensor:
        kv = self._get_kvs(prompt)
        model, _ = self._load_llm()
        inputs, attention_mask = inputs
        z = torch.zeros(
            1, kv[0][0].shape[-2]
        ).fill_(1).to("cuda")
        attention_mask = torch.cat((z, attention_mask,),dim=-1)
        inputs["attention_mask"] = attention_mask
        with torch.no_grad():
            q = model(**inputs, past_key_values=kv)
        return q["logits"][0, -1].softmax(-1)

    def _get_kvs(self, prompt: int) -> torch.Tensor:
        "Gets a key value vector for a prompt"
        self.feat_count[prompt] += 1
        if prompt in self.cache:
            return self.cache[prompt]
        
        if prompt in [a for a, _ in self.feat_count.most_common(self.CACHE)]:
            self.cache[prompt] = self.prompt_kvs[prompt].to("cuda")
            if len(self.cache) > self.CACHE:
                del self.cache[self.feat_count.most_common(self.CACHE + 1)[-1][0]]
        return self.prompt_kvs[prompt].to("cuda")


    def _classify(self, example: Tuple[str, torch.Tensor] , node_id: int) -> int:
        tree = self.state.tree
        feat = tree.feature[node_id]
        L, R = tree.children_left[node_id], tree.children_right[node_id]
        is_split_node = L != R
        prompt = feat // len(self.verb_tokenized)
        check = feat % len(self.verb_tokenized)
        if is_split_node:
            dist = self._run_one(example, prompt)
            v = dist[self.verb_tokenized[check]].item()
            next = L if v <= tree.threshold[node_id] else R
            return self._classify(example, next)
        else:
            ret = tree.value[node_id].argmax()
            return ret

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="rotten_tomatoes")
    parser.add_argument("--num_shots", type=int, default=128)
    parser.add_argument("--num_prompts", type=int, default=40)
    parser.add_argument("--max_leaf_nodes", type=int, default=40)
    parser.add_argument("--prompt_max_length", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--cache_size", type=int, default=3)
    parser.add_argument("--file_name", type=str, default="tree.pkl")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    args = parser.parse_args()

    dataset = datasets.load_dataset(args.dataset)
    tree = PromptTree(num_shots=args.num_shots,
                      num_prompts=args.num_prompts,
                      max_leaf_nodes=args.max_leaf_nodes,
                      prompt_max_length=args.prompt_max_length,
                      cache_size=args.cache_size,
                      seed=args.seed)
    dataset = datasets.load_dataset(args.dataset, "train")
    dataset = dataset.shuffle(seed=42)
    if args.do_train:
        tree.fit(dataset["train"].select(range(1000)),                       
                 batch_size=args.batch_size,)
        tree.save(args.file_name)
    else:
        tree.load(args.file_name)
    if args.do_test:
        tree.predict(dataset["test"])