import glob
import importlib
import json
import os
import sys


# use the current directory as the root
def run(path_raw_prompts: str = "./agent/prompts/raw", output_dir: str = "./agent/prompts/jsons") -> None:
    """Convert all python files in agent/prompts to json files in agent/prompts/jsons

    Python files are easiser to edit
    """
    python_to_json_prompt_by_id(path_raw_prompts=path_raw_prompts, prompt_id="", output_dir=output_dir)
    print("Done converting prompt python files to json")


def python_to_json_prompt_by_id(
    path_raw_prompts: str = "./agent/prompts/raw",
    prompt_id: str = "p_critique",
    output_dir: str = "./agent/prompts/jsons",
) -> None:
    path_raw_prompts = path_raw_prompts.strip()
    path_raw_prompts = os.path.normpath(path_raw_prompts)

    # Get all .py files in path_raw_prompts and its subdirectories
    for p_file in glob.glob(os.path.join(path_raw_prompts, "**", "*.py"), recursive=True):
        # check if the prompt_id is in the file; note, empty string => all prompts converted
        if prompt_id in p_file:
            # import the file as a module
            p_dir, p_basename = os.path.split(p_file)
            module_name = p_basename.replace(".py", "")
            module_path = p_dir.replace("/", ".")

            # Force reload if module was previously imported
            full_module_name = f"{module_path}.{module_name}"
            if full_module_name in sys.modules:
                del sys.modules[full_module_name]

            module = importlib.import_module(f"{module_path}.{module_name}")
            prompt = module.prompt

            # Delete the module from sys.modules
            del sys.modules[full_module_name]

            # save the prompt as a json file
            os.makedirs(output_dir, exist_ok=True)
            with open(f"{output_dir}/{module_name}.json", "w") as f:
                json.dump(prompt, f, indent=2)


if __name__ == "__main__":
    run()
