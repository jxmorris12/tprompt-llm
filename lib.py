from typing import Tuple

import datasets

def prepare_dataset(
  dataset: datasets.Dataset,
  column: str = "text",
  num_labels: int
) -> Tuple[List[str], List[str], List[str]]:
  """Prepares a dataset for Tree Prompting.

  Args:
    dataset (datasets.Dataset) - the dataset to use for tree construction
    column (str) - name of column to get text from, defaults to
      "text"
    num_prompts (int) - number of prompts to use for feature generation

  Returns:
    prompts (List(str)) - the list of prompts for feature generation
    data (List(str)) - list of data points that will be used w/ prompts
      to create features
    labels (List(str)) - the class labels used in prompt generation
  """
  raise NotImplementedException

def create_features(
  prompts: List[str],
  data: List[str],
  labels: List[str],
) -> List[List[int]]:
  raise NotImplementedException


def build_tree(
  features: List[List[int]]
) -> sklearn.tree.DecisionTreeClassifier:  
  raise NotImplementedException
