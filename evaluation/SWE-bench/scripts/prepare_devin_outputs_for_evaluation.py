"""
Script used to convert devin's output into the desired json format for evaluation on SWE-bench

Usage:
    python prepare_devin_outputs_for_evaluation.py <setting>
    <setting> can be "passed", "failed", "all"

Outputs:
    two json files under evaluation/SWE-bench/data/

"""

import json

# fetch devin's outputs into a json file for evaluation
import os
import sys

import requests
from tqdm import tqdm


def get_devin_eval_output(setting):
    repo_url = "CognitionAI/devin-swebench-results"
    folder_path = "output_diffs"

    base_url = "https://api.github.com/repos/"
    pass_api_url = f"{base_url}{repo_url}/contents/{folder_path}/pass"
    failed_api_url = f"{base_url}{repo_url}/contents/{folder_path}/fail"

    pass_files_info = []
    failed_files_info = []

    def get_files(api_url, subfolder_name, files_info):
        response = requests.get(api_url)
        if response.status_code == 200:
            contents = response.json()
            for item in tqdm(contents):
                if item["type"] == "file":
                    file_url = f"https://raw.githubusercontent.com/{repo_url}/main/{folder_path}/{subfolder_name}/{item['name']}"
                    file_content = requests.get(file_url).text
                    instance_id = item["name"][:-9]
                    model_name = "Devin"  # Update with actual model name
                    files_info.append(
                        {
                            "instance_id": instance_id,
                            "model_patch": file_content,
                            "model_name_or_path": model_name,
                            "pass_or_fail": subfolder_name,
                        }
                    )

    if setting == "passed" or setting == "all":
        get_files(pass_api_url, "pass", pass_files_info)
    if setting == "failed" or setting == "all":
        get_files(failed_api_url, "fail", failed_files_info)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "../data/devin/")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if setting == "passed" or setting == "all":
        with open(os.path.join(output_dir, "devin_swe_passed.json"), "w") as pass_file:
            json.dump(pass_files_info, pass_file, indent=4)

    if setting == "failed" or setting == "all":
        with open(os.path.join(output_dir, "devin_swe_failed.json"), "w") as fail_file:
            json.dump(failed_files_info, fail_file, indent=4)

    if setting == "all":
        merged_output = pass_files_info + failed_files_info
        with open(
            os.path.join(output_dir, "devin_swe_outputs.json"), "w"
        ) as merge_file:
            json.dump(merged_output, merge_file, indent=4)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <setting>")
        sys.exit(1)

    setting = sys.argv[1]
    get_devin_eval_output(setting)
