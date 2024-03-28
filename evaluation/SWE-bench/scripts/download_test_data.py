from datasets import load_dataset
import pandas as pd

dataset = load_dataset("princeton-nlp/SWE-bench_Lite_oracle")
test = dataset["test"].to_pandas()
test.to_json("data/processed/swe-bench-test.json", orient="records")
