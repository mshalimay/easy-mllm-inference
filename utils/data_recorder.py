import json
import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image

from browser_env.actions import Action, ActionTypes
from llms.providers.google.google_utils import google_count_tokens
from utils.timing_utils import timeit


class DataRecorder:
    def __init__(
        self, result_dir: str, config_file_list: list[str], test_config_base_dir: str, action_splitter: str = None
    ):
        self.result_dir = Path(result_dir)
        self.benchmark, self.category = self.parse_benchmark_domain(test_config_base_dir)
        self.action_splitter = action_splitter
        self.total_num_tasks = len(config_file_list)
        self.num_failed_executions = 0
        self.test_config_base_dir = test_config_base_dir

        # Initialize data structures
        self.observation_lens = {}
        self.token_counts = {}
        self.token_counts_per_agent = {}
        self.summary_data = {}
        self.data = {}
        self.failed_task_ids = set()
        self.unfinished_task_ids = {
            int(os.path.basename(config_file).split(".")[0]) for config_file in config_file_list
        }

        # Load existing data if resuming experiment
        self._load_existing_data()
        self._load_failed_tasks()

    def _load_failed_tasks(self):
        if (self.result_dir / "failed_tasks.txt").exists():
            with open(self.result_dir / "failed_tasks.txt", "r") as f:
                for line in f.readlines():
                    try:
                        self.failed_task_ids.add(int(line.strip()))
                    except ValueError:
                        continue

    def _load_existing_data(self):
        if (self.result_dir / "data.json").exists():
            with open(self.result_dir / "data.json", "r") as f:
                self.data = json.load(f)

        if (self.result_dir / "summary_data.csv").exists():
            self.summary_data = pd.read_csv(self.result_dir / "summary_data.csv", index_col="task_id")
            self.summary_data = self.summary_data.to_dict(orient="index")

    def log_token_counts(self, task_id: int, provider: str, api_response, agent_name=""):
        if provider == "openai":
            prompt_tokens = api_response.usage.prompt_tokens
            gen_tokens = api_response.usage.completion_tokens
        elif provider == "google":
            prompt_tokens = api_response.usage_metadata.prompt_token_count
            gen_tokens = api_response.usage_metadata.candidates_token_count
        else:
            raise NotImplementedError(f"Provider {provider} not implemented for logging token counts")

        self.token_counts[task_id]["full_prompt"].append(prompt_tokens)
        self.token_counts[task_id]["generation"].append(gen_tokens)

        if not agent_name:
            return

        if agent_name not in self.token_counts_per_agent[task_id]:
            self.token_counts_per_agent[task_id][agent_name] = {"full_prompt": [], "generation": []}

        self.token_counts_per_agent[task_id][agent_name]["full_prompt"].append(prompt_tokens)
        self.token_counts_per_agent[task_id][agent_name]["generation"].append(gen_tokens)

    @timeit(custom_name="DATA:log_observation_len")
    def log_observation_len(self, trajectory, task_id, model, provider, tokenizer=None, asynchronous=False):
        """Log observation length of current step"""
        state_info = trajectory[-1]
        obs = state_info["observation"]

        if "text" in obs:
            self.observation_lens[task_id]["text_raw"].append(len(obs["text"]))
            if provider == "google":
                self.observation_lens[task_id]["text_tokenized"].append(
                    google_count_tokens(model=model, lm_input=obs["text"], asynchronous=asynchronous)
                )
            else:
                self.observation_lens[task_id]["text_tokenized"].append(len(tokenizer.encode(obs["text"])))

        if "image" in obs:
            if provider == "google":
                goog_img = Image.fromarray(obs["image"])
                self.observation_lens[task_id]["img_tokenized"].append(
                    google_count_tokens(model=model, lm_input=goog_img, asynchronous=asynchronous)
                )
            elif provider == "openai":
                pass  # Currently tiktoken does not support counting tokens for images
            elif provider == "huggingface":
                raise NotImplementedError("Huggingface provider not implemented")

    def initialize_task(self, task_id: int, sites: List[str]) -> Dict:
        """Initialize data recording structures for a new task"""
        self.observation_lens[task_id] = {
            "text_raw": [],
            "text_tokenized": [],
            "img_tokenized": [],
        }

        self.token_counts[task_id] = {"full_prompt": [], "generation": []}

        self.token_counts_per_agent[task_id] = {}

        actions = {
            "action_types": [],
            "action_utterances": [],
            "parsed_action_sequence": [],
            "num_actions": 0,
            "num_invalid_actions": 0,
        }

        self.data[task_id] = {
            "actions": actions,
            "elapsed_time": np.nan,
            "score": np.nan,
            "obs_lens": self.observation_lens[task_id],
            "benchmark": self.benchmark,
            "category": self.category,
            "sites": sites,
        }

        self.summary_data[task_id] = {}

        return {"action_history": ["None"], "observation_lens": self.observation_lens}

    def parse_action(self, action_str: str, action_splitter: str) -> str:
        pattern = rf"{action_splitter}((.|\n)*?){action_splitter}"
        match = re.search(pattern, action_str)
        if match:
            return match.group(1).strip()
        else:
            return "N/A"

    @timeit(custom_name="DATA:record_action")
    def record_action(self, task_id: int, action: Action):
        self.data[task_id]["actions"]["action_types"].append(action["action_type"].name)
        self.data[task_id]["actions"]["action_utterances"].append(action["raw_prediction"])
        self.data[task_id]["actions"]["parsed_action_sequence"].append(
            self.parse_action(action["raw_prediction"], self.action_splitter)
        )
        if action["action_type"] == ActionTypes.NONE:
            self.data[task_id]["actions"]["num_invalid_actions"] += 1

    def update_save_data(self, task_id: int, score: float, elapsed_time: float, num_actions: int):
        """Update all data and summary statistics for a task"""
        self.compute_token_stats(task_id)

        # Record score, number of actions, elapsed time
        self.summary_data[task_id]["score"] = score
        self.summary_data[task_id]["num_actions"] = num_actions
        self.summary_data[task_id]["elapsed_time"] = elapsed_time
        self.data[task_id]["score"] = score
        self.data[task_id]["elapsed_time"] = elapsed_time
        self.data[task_id]["actions"]["num_actions"] = num_actions
        self.save_to_disk()

    def compute_token_stats(self, task_id: int):
        """Compute statistics for observation lengths"""
        keys_to_labels = {
            "text_raw": "text_obs_len_raw",
            "text_tokenized": "text_obs_len_tok",
            "img_tokenized": "img_obs_len_tok",
        }

        for key, label in keys_to_labels.items():
            stats = self._calculate_stats(self.observation_lens[task_id][key])
            self.summary_data[task_id][f"sum_{label}"] = stats["sum"]
            self.summary_data[task_id][f"avg_{label}"] = stats["avg"]
            self.summary_data[task_id][f"max_{label}"] = stats["max"]
            self.summary_data[task_id][f"min_{label}"] = stats["min"]

        keys_to_labels = {"full_prompt": "prompt_tokens", "generation": "gen_tokens"}

        for key, label in keys_to_labels.items():
            stats = self._calculate_stats(self.token_counts[task_id][key])
            self.summary_data[task_id][f"sum_{label}"] = stats["sum"]
            self.summary_data[task_id][f"avg_{label}"] = stats["avg"]
            self.summary_data[task_id][f"max_{label}"] = stats["max"]
            self.summary_data[task_id][f"min_{label}"] = stats["min"]

        for agent_name, counts in self.token_counts_per_agent[task_id].items():
            for key, label in keys_to_labels.items():
                stats = self._calculate_stats(counts[key])
                self.summary_data[task_id][f"sum_{agent_name}_{label}"] = stats["sum"]
                self.summary_data[task_id][f"avg_{agent_name}_{label}"] = stats["avg"]
                self.summary_data[task_id][f"max_{agent_name}_{label}"] = stats["max"]
                self.summary_data[task_id][f"min_{agent_name}_{label}"] = stats["min"]

    @timeit(custom_name="DATA:save_execution_summary")
    def save_execution_summary(
        self,
        total_time: float,
        provider: str | list[str],
    ):
        """Save summary of the entire execution"""
        execution_data = [
            {
                "benchmark": self.benchmark,
                "category": self.category,
                "n_tasks": self.total_num_tasks,
                "n_failed_executions": self.num_failed_executions,
                "total_time": total_time,
                "provider": provider,
            }
        ]

        # Save to temp file + rename to avoid race conditions
        dest_file_csv = self.result_dir / "execution_data.csv"
        tmp_csv_file = dest_file_csv.with_suffix(".tmp")
        pd.DataFrame(execution_data).to_csv(tmp_csv_file, index=False)
        tmp_csv_file.rename(dest_file_csv)

    @timeit(custom_name="DATA:save_to_disk")
    def save_to_disk(self):
        """Save current data to disk"""
        # Save to temp file + rename to avoid race conditions

        dest_file_csv = self.result_dir / "summary_data.csv"
        tmp_csv_file = dest_file_csv.with_suffix(".tmp")
        df = pd.DataFrame.from_dict(self.summary_data, orient="index")
        df.to_csv(tmp_csv_file, index=True, index_label="task_id")
        tmp_csv_file.rename(dest_file_csv)

        tmp_json_file = self.result_dir / "data.json.tmp"
        with open(tmp_json_file, "w") as f:
            json.dump(self.data, f, indent=4)
        tmp_json_file.rename(self.result_dir / "data.json")

    @staticmethod
    def _calculate_stats(data: list) -> dict:
        """Calculate basic statistics for a list of numbers"""
        try:
            stats = {"sum": sum(data), "avg": sum(data) / len(data), "max": max(data), "min": min(data)}
        except ZeroDivisionError:
            stats = {"sum": None, "avg": None, "max": None, "min": None}
        return stats

    def parse_benchmark_domain(self, test_config_base_dir: str) -> tuple[str | None, str | None]:
        match = re.search(r"config_files/(\w+)/test_(\w+)", test_config_base_dir)
        if match:
            return match.group(1), match.group(2)
        return None, None

    def _update_unfinished_failed_tasks(self, task_id: int, task_success: bool, task_set: set, file_path: Path):
        initial_len = len(task_set)
        if task_success:
            task_set.discard(task_id)
        else:
            task_set.add(task_id)

        if initial_len != len(task_set):
            with open(file_path, "w") as f:
                f.write(self.test_config_base_dir + "\n")
                for task_id in sorted(task_set):
                    f.write(f"{task_id}\n")

    def update_unfinished_failed_tasks(self, task_id: int, task_success: bool):
        self._update_unfinished_failed_tasks(
            task_id, task_success, self.failed_task_ids, self.result_dir / "failed_tasks.txt"
        )
        self._update_unfinished_failed_tasks(
            task_id, task_success, self.unfinished_task_ids, self.result_dir / "unfinished_tasks.txt"
        )

    def get_scores(self) -> list[float]:
        return [
            self.summary_data[task_id]["score"]
            for task_id in self.summary_data
            if "score" in self.summary_data[task_id]
        ]
