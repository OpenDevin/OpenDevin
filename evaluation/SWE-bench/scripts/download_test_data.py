import pandas as pd
from datasets import load_dataset

dataset = load_dataset("princeton-nlp/SWE-bench")
test = dataset["test"].to_pandas()
test.to_json("data/processed/swe-bench-test.json", orient="records")
