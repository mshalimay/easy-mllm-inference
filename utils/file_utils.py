import hashlib
import json
import os
import re
import shutil
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dateutil import parser
from joblib import Parallel, delayed
from pathvalidate import is_valid_filename


# ===============================================================================
# LINK: General file utils
# ===============================================================================
def is_empty(variable: Any) -> bool:
    if variable is None:
        return True
    if isinstance(variable, np.ndarray):
        return variable.size == 0  # An empty array
    return not variable  # For other types (empty string, empty list, etc.)


def count_files(
    dir_path: str, filename=None, starts_with=None, ends_with=None, re_pattern=None, exclude_dirs: list[str] = []
):
    """Check if at least `n` files match a given prefix or suffix in a directory, including all nested subdirs."""
    if filename and (starts_with or ends_with or re_pattern):
        raise Warning("filename provided, ignoring starts_with, ends_with, and re_pattern")

    if not Path(dir_path).exists():
        return 0

    if Path(dir_path).is_file():
        files = [Path(dir_path)]
    else:
        files = [f for f in Path(dir_path).rglob("*") if f.is_file() and not any(d in str(f) for d in exclude_dirs)]

    # If filename provided, return count of exact matches
    if filename is not None:
        return sum(bool(re.match(filename, path.name)) for path in files)

    # If starts_with and ends_with are provided and are the same, return count of exact matches
    if all([starts_with, ends_with]) and starts_with == ends_with:
        return sum(bool(re.match(starts_with, path.name)) for path in files)

    # If re_pattern is provided, return count of matches to the regex pattern
    if re_pattern is not None:
        return sum(bool(re.match(re_pattern, path.name)) for path in files)

    # Match files that start with `starts_with`, end with `ends_with`
    pattern = r"^"  # Ensure match starts at the beginning

    if starts_with is not None:
        pattern += re.escape(starts_with)  # Exact prefix match
    else:
        pattern += r".*"  # Allow anything at the start

    pattern += r".*"  # Allow anything in the middle

    if ends_with is not None:
        pattern += re.escape(ends_with) + r"$"  # Exact suffix match
    else:
        pattern += r".*$"  # Allow anything at the end

    regex = re.compile(pattern)

    return sum(bool(regex.match(path.name)) for path in files)


def parse_datetime(date_string: str) -> datetime | None:
    """
    Converts a string into a datetime object, automatically detecting the format.

    Args:
        date_string (str): The datetime string to parse.

    Returns:
        datetime: A datetime object if parsing is successful.
        None: If parsing fails.
    """
    try:
        return parser.parse(date_string)
    except (ValueError, TypeError):
        return None  # Return None if parsing fails


def flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ":") -> dict[str, Any]:
    """
    Flatten a nested dictionary by concatenating parent keys with child keys.
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if hasattr(v, "to_dict"):  # Handle objects with to_dict method
            v = v.to_dict()

        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def make_json_friendly(obj, ignore_order=True):
    """
    Recursively convert non-JSON-serializable types (like sets) into
    JSON-serializable equivalents.
    """
    if isinstance(obj, set):
        # Convert set to a sorted list for deterministic output
        return sorted(obj)

    elif isinstance(obj, tuple):
        if ignore_order:
            try:
                # Convert tuple to a list (JSON can handle lists but not tuples distinctly)
                return [make_json_friendly(e) for e in sorted(obj)]
            except:
                return [make_json_friendly(e) for e in obj]
        else:
            return [make_json_friendly(e) for e in obj]

    elif isinstance(obj, list):
        if ignore_order:
            try:
                return [make_json_friendly(e) for e in sorted(obj)]
            except:
                return [make_json_friendly(e) for e in obj]
        else:
            return [make_json_friendly(e) for e in obj]

    elif isinstance(obj, dict):
        # Recursively convert values; sort keys for consistency
        return {k: make_json_friendly(v) for k, v in sorted(obj.items(), key=lambda x: x[0])}
    else:
        # For anything else that JSON can already handle (str, int, float, bool, None, etc.)
        return obj


def stable_json_hash(obj) -> str:
    """
    Convert `obj` into a JSON-serializable canonical form, then produce an MD5 hash
    of the sorted-keys JSON dump.
    """
    normalized_obj = make_json_friendly(obj)
    serialized = json.dumps(normalized_obj, sort_keys=True)
    return hashlib.md5(serialized.encode("utf-8")).hexdigest()


def strip_path(path: str, relative_to: str):
    try:
        return str(Path(path).relative_to(relative_to))
    except:
        return path


def is_relative_to(path: str, relative_to: str):
    try:
        Path(path).relative_to(relative_to)
        return True
    except:
        return False


def rejoin_path(path: str, relative_to: str):
    try:
        return str(Path(path).joinpath(relative_to))
    except:
        return path


def get_common_paths(paths, relative_to=""):
    if len(paths) == 1:
        return [paths[0]], [paths[0]]

    elif len(paths) == 0:
        return [], []

    # TODO: stress test this function
    if relative_to:
        stripped_paths = [strip_path(p, relative_to) for p in paths if is_relative_to(p, relative_to)]
    else:
        stripped_paths = paths

    sorted_paths = sorted(stripped_paths)
    common_paths = set()

    # Obs.: quadratic complexity; can be improved
    max_path_len = 0
    min_path_len = float("inf")
    for i in range(len(sorted_paths) - 1):
        for j in range(i + 1, len(sorted_paths)):
            common_path = os.path.commonpath([sorted_paths[i], sorted_paths[j]])
            if common_path:
                common_paths.add(common_path)
                max_path_len = max(max_path_len, len(Path(common_path).parts))
                min_path_len = min(min_path_len, len(Path(common_path).parts))

    longest_common_paths = [p for p in common_paths if len(Path(p).parts) == max_path_len]
    shortest_common_paths = [p for p in common_paths if len(Path(p).parts) == min_path_len]

    if relative_to:
        longest_common_paths = [Path(relative_to) / p for p in longest_common_paths]
        shortest_common_paths = [Path(relative_to) / p for p in shortest_common_paths]
        common_paths = [Path(relative_to) / p for p in common_paths]

    return (
        [str(p) for p in longest_common_paths],
        [str(p) for p in shortest_common_paths],
        [str(p) for p in common_paths],
    )


def remove_empty_dirs(path, exclude_dirs: list[str] = []):
    """Remove all subdirs of `path` that are empty.
    Obs.: An empty subdir is defined as a subdir that does not contain any files in it or in its subdirs."""

    if not Path(path).exists():
        return

    dirs = sorted([d for d in Path(path).rglob("*") if d.is_dir()], reverse=True)

    for dir_path in dirs:
        if any(exclude_dir in str(dir_path) for exclude_dir in exclude_dirs):
            continue

        if dir_path.is_dir() and not any(dir_path.iterdir()):
            dir_path.rmdir()
            print(f"Removed empty directory: {dir_path}")

    if not any(Path(path).iterdir()):
        Path(path).rmdir()
        print(f"Removed empty directory: {path}")


def get_dirs_with_file(dir_path: str, filename: str, exclude_dirs: list[str] = []) -> list[Path]:
    """
    Get all directories in `dir_path` that contain a file with the given `filename`.
    """
    key_files = find_files(dir_path, filename, upwards=False, downwards=True)
    return [Path(f).parent for f in key_files if not any(d in str(f) for d in exclude_dirs)]


# ===============================================================================
# LINK File/Dir content comparison
# ===============================================================================


def get_hash_files_map(
    dir_path: str | Path = "",
    files: list[Path] = [],
    n_jobs: int = 8,
    exclude_dirs: list[str] = [],
) -> dict[str, Path]:
    if files and dir_path:
        print(f"Warning: `files` and `dir_path` are both provided. dir_path will be ignored.")

    if not files and not dir_path:
        raise ValueError("Either `files` or `dir_path` must be provided.")

    if not files:
        dir_path = Path(dir_path)
        if not dir_path.exists():
            return []
        else:
            files = [f for f in dir_path.rglob("**/*") if f.is_file() and not any(d in str(f) for d in exclude_dirs)]

    if n_jobs == 1:
        results = {get_file_hash(f): f for f in files}

    else:
        hashes = Parallel(n_jobs=n_jobs)(delayed(get_file_hash)(f) for f in files)
        results = {h: f for h, f in zip(hashes, files)}

    return results


def identical_dir_content(
    dir1: str | Path, dir2: str | Path, ignore_dir_struct: bool = False, exclude_dirs: list[str] = []
) -> tuple[bool, dict[str, Path], dict[str, Path]]:
    dir1, dir2 = Path(dir1), Path(dir2)
    if not dir1.exists() or not dir2.exists():
        return False, {}, {}

    if not dir1.is_dir() or not dir2.is_dir():
        return False, {}, {}

    all_files_1 = [f for f in Path(dir1).rglob("*") if f.is_file() and not any(d in str(f) for d in exclude_dirs)]
    all_files_2 = [f for f in Path(dir2).rglob("*") if f.is_file() and not any(d in str(f) for d in exclude_dirs)]

    if len(all_files_1) != len(all_files_2):
        return False, {}, {}

    hashes_files_1 = get_hash_files_map(dir1, files=all_files_1)
    hashes_files_2 = get_hash_files_map(dir2, files=all_files_2)

    if ignore_dir_struct:
        return (
            sorted(hashes_files_1.keys()) == sorted(hashes_files_2.keys()),
            hashes_files_1,
            hashes_files_2,
        )
    else:
        hash_rel_paths_1 = {h: Path(f).relative_to(dir1) for h, f in hashes_files_1.items()}
        hash_rel_paths_2 = {h: Path(f).relative_to(dir2) for h, f in hashes_files_2.items()}
        return hash_rel_paths_1 == hash_rel_paths_2, hashes_files_1, hashes_files_2


def count_dirs_duplicates(
    dir1: str | Path,
    dir2: str | Path,
    ignore_dir_struct: bool = False,
    exclude_dirs: list[str] = [],
) -> tuple[bool, dict[str, Path], dict[str, Path]]:
    # Normalize params
    dir1, dir2 = Path(dir1), Path(dir2)
    if not dir1.exists() or not dir2.exists():
        return False, {}, {}

    if not dir1.is_dir() or not dir2.is_dir():
        return False, {}, {}

    all_files_1 = [f for f in Path(dir1).rglob("*") if f.is_file() and not any(d in str(f) for d in exclude_dirs)]
    all_files_2 = [f for f in Path(dir2).rglob("*") if f.is_file() and not any(d in str(f) for d in exclude_dirs)]

    hashes_files_1 = get_hash_files_map(dir1, files=all_files_1)
    hashes_files_2 = get_hash_files_map(dir2, files=all_files_2)

    if ignore_dir_struct:
        num_duplicates = 0
        for hash in hashes_files_1:
            if hash in hashes_files_2:
                num_duplicates += 1
        return num_duplicates, hashes_files_1, hashes_files_2

    hash_rel_paths_1 = {h: Path(f).relative_to(dir1) for h, f in hashes_files_1.items()}
    hash_rel_paths_2 = {h: Path(f).relative_to(dir2) for h, f in hashes_files_2.items()}

    num_duplicates = 0
    for hash, file in hash_rel_paths_1.items():
        if hash in hash_rel_paths_2 and hash_rel_paths_1[hash] == hash_rel_paths_2[hash]:
            num_duplicates += 1

    return num_duplicates, hashes_files_1, hashes_files_2


def identical_file_content(file1: Path | str, file2: Path | str) -> bool:
    """
    Returns True if file1 and file2 have identical contents by comparing MD5 hashes.
    """
    file1 = Path(file1)
    file2 = Path(file2)
    if not file1.exists() or not file2.exists():
        return False

    if not file1.is_file() or not file2.is_file():
        return False

    return hashlib.md5(file1.read_bytes()).hexdigest() == hashlib.md5(file2.read_bytes()).hexdigest()


def resolve_path_conflict(path: Path | str, int_suffix: bool = False) -> Path:
    orig_path = Path(str(path))  # Force copy to make sure dont change the original Path object

    # Obs.: using `a`, `b`, `c`, etc. to resolve conflicts as some experiments use numerical IDs to identify tasks.

    # Generate lowercase letters a-z, then aa, ab, ac, etc.
    def get_letter_suffix(n: int) -> str:
        if n < 26:
            return chr(97 + n)  # 97 is ASCII for 'a'
        else:
            return chr(97 + (n // 26 - 1)) + chr(97 + (n % 26))

    idx, new_path = 0, Path(str(orig_path))  # Force copy of object
    while new_path.exists():
        idx += 1
        annotation = f"_{get_letter_suffix(idx)}" if not int_suffix else f"_{idx}"
        new_path = Path(
            add_annotation_to_path(
                path=orig_path,
                annotation=annotation,
                add_to_end=True,
            )
        )

    return new_path


class SanityChecker:
    def __init__(self, num_processes: int = 4):
        self.hash_to_file_orig = None
        self.hash_to_file_new = None
        self.num_processes = num_processes

    def set_original_hashes(self, dir_path: str, exclude_dirs: list[str] = []):
        self.hash_to_file_orig = get_hash_files_map(dir_path, exclude_dirs=exclude_dirs, n_jobs=self.num_processes)

    def set_new_hashes(self, dir_path: str, exclude_dirs: list[str] = []):
        self.hash_to_file_new = get_hash_files_map(dir_path, exclude_dirs=exclude_dirs, n_jobs=self.num_processes)

    def sanity_check(self, orig_hashes="", new_hashes=""):
        if orig_hashes:
            self.set_original_hashes(orig_hashes)

        if new_hashes:
            self.set_new_hashes(new_hashes)

        if not self.hash_to_file_orig or not self.hash_to_file_new:
            print(f"Hashes not set for comparison.")
            return

        print("\n\n======== Sanity check ========\n\n")

        print("\nHashes IN ORIGINAL DIR but NOT in NEW DIR:\n")
        for hash in self.hash_to_file_orig:
            if hash not in self.hash_to_file_new:
                filename = self.hash_to_file_orig[hash].name
                if "attempted_tasks.txt" in filename:
                    continue
                print(self.hash_to_file_orig[hash])

        print("\nHashes IN NEW DIR but NOT in ORIGINAL DIR:\n")
        for hash in self.hash_to_file_new:
            if hash not in self.hash_to_file_orig:
                print(self.hash_to_file_new[hash])


# ===============================================================================
# copy/move
# ===============================================================================


def copy_move_optional_ovewrite(
    src: str | Path, dst: str | Path, mode: str, follow_symlinks: bool = True, overwrite: bool = False
):
    """
    A replacement for shutil.copy2 that first checks if the destination file
    already exists and has identical contents. If so, we skip copying.
    """
    dst = Path(dst)

    if identical_file_content(src, dst):
        print(f"Skipping copying because identical: `{src}` -> `{dst}`")
        if mode == "move":
            Path(src).unlink(missing_ok=True)
        return str(dst)  # Return the existing path without copying

    # If overwrite, simple copy
    if overwrite:
        final_dst = shutil.copy2(src, dst, follow_symlinks=follow_symlinks)
        if mode == "move":
            Path(src).unlink(missing_ok=True)
        return final_dst

    # Check for conflicts and resolve them
    new_dst = resolve_path_conflict(dst)

    # If conflicts and no overwrite, print warning
    if str(new_dst) != str(dst):
        print(f"Path `{dst}` already exists and files have different content, copying to: `{new_dst}` instead")

    # move / copy (new_dst is guaranteed to be unique at this point)
    # Make sure destination exists
    new_dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        return shutil.copy2(src, new_dst, follow_symlinks=follow_symlinks)
    else:
        return shutil.move(src, new_dst)


def copy_move(
    src: str | Path,
    dst: str | Path,
    mode: str,
    overwrite_file: bool = False,
    merge_dirs: bool = True,
    copy_move_only_new: bool = True,
    exclude_strs: list[str] = [],
    ignore_dir_struct: bool = True,
) -> str:
    """
    Copies or moves a file or directory from `src` to `dst`, with flexible handling of
    conflicting paths and identical content checks.

    Args:
        src (str | Path):
            The source file or directory to copy or move.
        dst (str | Path):
            The target path where data will be copied or moved.
        mode (str):
            Operation mode. Must be either `"copy"` or `"move"`.

        overwrite_file (bool, optional):
            If True, individual files with the same destination path are overwritten when copying.
            If False, conflicting files are renamed by appending a suffix, or the copy is skipped if
            their content is identical.

        merge_dirs (bool, optional):
            If True, move/copy into an existing directory at `dst`. If false, move/copy to a new directory.
            Obs.: If merging to an existing dir, and it contains files with same contents as the source,
            the behavior for files (overwrite or not) follows `overwrite_file`.

        copy_move_only_new (bool, optional):
            If True, only copy/move files that are not duplicates.
            If `merge_dirs=False`, this means the new created directory will only contain the non-duplicates.

    Returns:
        str:
            The final destination path (possibly with a suffix if a conflict occurred),
            or an empty string if the operation was skipped (e.g., incompatible file/directory
            types or identical content with a skip scenario).

    Raises:
        ValueError:
            If `mode` is not `"copy"` or `"move"`.
    """
    # Check if valid mode
    if mode not in ["copy", "move"]:
        raise ValueError(f"Invalid mode: {mode}")

    # Normalize to Path objects
    src, dst = Path(src), Path(dst)

    # If src to copy from does not exist, skip
    if not src.exists():
        print(f"Source path `{src}` does not exist, skipping {mode}.")
        return ""

    # If copying/moving to itself, skip
    if str(src) == str(dst):
        return str(dst)

    # CASE 0: If copy/move of dir to a file, raise error
    if src.is_dir() and dst.is_file():
        raise ValueError(f"Trying to {mode} a directory to a file.")

    # CASE 1: copy/move file to file or file to dir
    if src.is_file():
        return copy_move_optional_ovewrite(src=src, dst=dst, mode=mode, overwrite=overwrite_file)

    # CASE 2: copy/move of dir to dir

    # Obs.: Code below is a bit inneficient, but more readable

    # Maps each source directory to its *resolved* destination directory
    dirs_to_dstdirs = {}

    # Go through each directory in src and compute a destination for it, resolving conflicts if not `merge_dirs`.
    # If src's `dirA` relative path in destination is changed due to conflicts, then all nested dirs of dirA have their
    # relative paths changed as well.
    top_level_dst = dst if merge_dirs else resolve_path_conflict(dst)
    dirs_to_dstdirs[src] = top_level_dst

    dirs_to_process = [src]
    while dirs_to_process:
        # Get the directory to process
        cur_dir = dirs_to_process.pop()

        # Get the destination for the directory being processed
        cur_dst = dirs_to_dstdirs[cur_dir]

        # Go through each subdir of the directory being processed
        for subdir in cur_dir.iterdir():
            if not subdir.is_dir():
                continue

            # Compute the subdir destination taking into account if parent had their relative path renamed due to conflicts.
            sub_dst = cur_dst / subdir.name

            # If we do NOT merge directories, resolve any conflicts for the subdir.
            if not merge_dirs:
                sub_dst = resolve_path_conflict(sub_dst)

            # Store subdir -> subdir destination in dictionary so the next iteration knows where subdir goes
            dirs_to_dstdirs[subdir] = sub_dst

            # Push subdir into the stack so we later process its children
            dirs_to_process.append(subdir)

    # If only copying/moving new files, get the subset of files to copy/move
    copy_subset = set()
    if copy_move_only_new:
        hash_files_src = get_hash_files_map(src)
        hash_files_dst = get_hash_files_map(dst)
        if ignore_dir_struct:
            # If ignoring dir struct, any identical files are labeled as duplicates
            copy_subset = {f for hash, f in hash_files_src.items() if hash not in hash_files_dst}
        else:
            # If not ignoring dir struct, only identical files in the same relative path are considered duplicates
            copy_subset = {
                f
                for hash, f in hash_files_src.items()
                if (
                    hash not in hash_files_dst  # file is not duplicate
                    or dirs_to_dstdirs[f.parent] != hash_files_dst[hash].parent  # relative dir struc differs
                )
            }

        # If only duplicates, skip
        if len(copy_subset) == 0:
            print(f"No new files to copy from `{src}`. Skipping `{mode}`.")
            return ""

    for dir, dst_dir in dirs_to_dstdirs.items():
        for f in dir.iterdir():
            if not f.is_file():
                continue

            # If duplicate file and moving only new, skip
            if copy_subset and f not in copy_subset:
                # If `move`, remove the file from src
                Path(f).unlink(missing_ok=True) if mode == "move" else None
                continue

            # If file matches any exclude string, skip
            if any(exclude_str in str(f) for exclude_str in exclude_strs):
                continue

            # Make sure dst directory exists
            dst_dir.mkdir(parents=True, exist_ok=True)

            # Get file path in the destination
            dst_path = dst_dir / f.name
            copy_move_optional_ovewrite(src=f, dst=dst_path, mode=mode, overwrite=overwrite_file)

    if "move" in mode:
        remove_empty_dirs(src)

    return dirs_to_dstdirs[src]


# ===============================================================================
# Search and count files / dirs
# ===============================================================================


def contains_file(dir_path: str, filename: str, num_files: int = 1):
    return count_files(dir_path, filename=filename) >= num_files


def contains_dir(dir_path: str, dirname: str) -> bool:
    search_path = Path(dir_path).resolve()
    return any(dirname in str(p) for p in search_path.rglob("*"))


def find_files(start_dir: str | Path, filename: str, upwards: bool = False, downwards: bool = True) -> list[str]:
    """
    Search for files named `filename` upward and downwards from `start_dir`.
    """
    files = []

    # Strip leading slashes from filename
    filename = filename.lstrip("/")

    # Search upwards
    if upwards:
        current_dir = Path(start_dir).resolve()
        while True:
            candidate = current_dir / filename
            if candidate.exists():
                files.append(str(candidate))
            if current_dir == current_dir.parent:
                # Reached the filesystem root
                break
            current_dir = current_dir.parent

    # Search downwards
    if downwards:
        files.extend([str(p) for p in Path(start_dir).rglob(filename)])

    return files


def is_bottom_level_dir(dir_path: str):
    return not any(p.is_dir() for p in Path(dir_path).iterdir())


def update_json(file_path: str, data: dict[str, Any], indent: int = 4) -> None:
    # TODO: make it recursive to update nested json

    # If no previous json data, just dump new data
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump(data, f, indent=indent)
        return

    # If previous json data exists, update it with new data, and dump
    with open(file_path, "r") as f:
        existing_data = json.load(f)

    # Converts any keys that are numbers to strings to correctly update the json file
    normalized_json = {num_to_str(k): v for k, v in data.items()}

    existing_data.update(normalized_json)
    with open(file_path, "w") as f:
        json.dump(existing_data, f, indent=indent)


def is_valid_path_str(path: str) -> bool:
    try:
        path_obj = Path(path)

        # Check for cases like:
        # "a/b//c" -> False; dir1/dir/dir3/.html -> False
        parts = path_obj.parts[1:]
        if parts and any(part.strip("/") == "" for part in parts):
            return False

        # Check for specific char not allowed in filenames
        if not is_valid_filename(path_obj.name):
            return False

        return True
    except:
        return False


def add_annotation_to_path(path: str | Path, annotation: str, add_to_end: bool = True) -> str:
    path_obj = Path(str(path))  # Force copy to make sure not changing the original path object

    parent_dir = str(path_obj.parent)
    filename_no_ext = str(path_obj.stem)
    ext = str(path_obj.suffix)

    if add_to_end:
        new_filename = f"{filename_no_ext}{annotation}{ext}"
    else:
        new_filename = f"{annotation}{filename_no_ext}{ext}"

    new_full_path = f"{parent_dir}/{new_filename}"

    # Check if new_path ends with invalid format
    path_obj = Path(new_full_path)
    if not is_valid_path_str(new_full_path):
        raise ValueError(f"New path is invalid: {new_full_path}. Please revise the given annotation.")

    return new_full_path


def get_args(dir_path: str, args_file_name: str = "args.json") -> dict:
    if not os.path.exists(dir_path):
        return {}

    ext = Path(dir_path).suffix
    if ext == ".json":
        with open(dir_path, "r") as f:
            return json.load(f)

    elif ext == "":
        with open(os.path.join(dir_path, args_file_name), "r") as f:
            return json.load(f)
    else:
        return {}


def num_to_str(num):
    if isinstance(num, str):
        return num

    try:
        return str(num)
    except:
        return num


def get_attribute_from_dict(key_path: str, dict_data: dict = {}, json_path: str = "", delimiter: str = ":"):
    if not json_path and not dict_data:
        raise ValueError("Either dict_data or json_path must be provided")

    if dict_data and json_path:
        print("Warning: dict_data and json_path provided, using dict_data")
        json_path = ""

    if json_path:
        with open(json_path, "r") as f:
            dict_data = json.load(f)

    keys = key_path.split(delimiter)
    data = dict_data
    for key in keys:
        data = data.get(key, "")
        if data == "":
            return None
    return data


def get_file_hash(file_path: str) -> str:
    return hashlib.md5(file_path.read_bytes()).hexdigest()


# ===============================================================================
# Almost general - requires some specific format in logs, args, etc
# ===============================================================================
def get_log_file(dir_path: str, get_more_recent: bool = True) -> str:
    direct_log_file = Path(dir_path) / "log.txt"
    if direct_log_file.exists():
        return str(direct_log_file)

    log_file_paths = Path(dir_path) / "log_files.txt"
    if not log_file_paths.exists():
        return ""

    with open(log_file_paths, "r") as f:
        lines = reversed(f.readlines()) if get_more_recent else f.readlines()
        for line in lines:
            log_file_path = line.strip()
            log_file_path = Path(re.sub("'", "", log_file_path))
            if log_file_path.exists():
                return str(log_file_path)

    return ""


def get_data_from_log_file(exec_dir_path: str, get_last: bool = False, re_pattern: str = ""):
    log_file_path = get_log_file(exec_dir_path)
    if not log_file_path:
        return None

    with open(log_file_path, "r") as f:
        lines = reversed(f.readlines()) if get_last else f.readlines()
        for line in lines:
            match = re.search(re_pattern, line.strip())
            if match:
                return match.group(0)
    return None


def extract_datetime_from_log_file(
    exec_dir_path: str, get_last: bool = False, re_pattern: str = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
):
    datetime_str = get_data_from_log_file(exec_dir_path=exec_dir_path, get_last=get_last, re_pattern=re_pattern)

    if not datetime_str:
        return None
    return parse_datetime(datetime_str)


def is_finished_execution(dir_path: str, max_time_diff_minutes: int = 10) -> bool:
    # If log file contains "Total test time:", then it is finished
    re_pattern = r"Total test time"
    match = get_data_from_log_file(dir_path, get_last=True, re_pattern=re_pattern)
    if match:
        return True

    # If log file does not contain "Total test time:", then check for last datetime in log file
    datetime_now = datetime.now()
    datetime_str = extract_datetime_from_log_file(exec_dir_path=dir_path, get_last=True)
    if datetime_str:
        # If last time in the log is more than max_time_diff_minutes old, then it is finished
        time_diff_minutes = (datetime_now - datetime_str).total_seconds() / 60
        return time_diff_minutes > max_time_diff_minutes

    # Default to not finished for safety
    return False


def extract_run_datetime(exec_dir_path: str):
    # Try to get pattern from directory name
    pattern = r"\d{4}-\d{2}-\d{2}-\d{4}"
    match = re.search(pattern, exec_dir_path)

    if match:
        datetime_str = match.group(0)
        # Parse the datetime string
        return parse_datetime(datetime_str)

    # Try to get datetime from log file
    datetime_str = extract_datetime_from_log_file(exec_dir_path=exec_dir_path, get_last=False)
    if datetime_str:
        return parse_datetime(datetime_str)

    return None


# ===============================================================================
# Enviroment specific (V)WA
# ===============================================================================
def get_task_id_from_file_path(file_path: str | Path) -> str:
    """
    Get the task id from a file path.
    The task id is the last integer that appears before the file extension (which can be empty, e.g. if `file_path` is a folder).
    Args:
        file_path (str): The path to the file or folder.

    Returns:
        str: The task id if found, otherwise an empty string.

    Examples:
        - local/experiments_1/domain_2/task234.json -> 234
        - tasks1/render_23_failed.html -> 23
        - tasks1/render_23 -> 23
        - tasks1/render_failed.html -> ""
    """
    file_path_obj = Path(file_path)
    ext = file_path_obj.suffix
    # Gets the LAST integer that appear before the file extension (which can be empty, for e.g. if file_path is a folder)
    # e.g: tasks1/render_23_failed.html -> 23
    match = re.search(rf".*?(\d+)[^0-9]*{ext}$", str(file_path_obj))
    if match:
        return match.group(1)
    else:
        return ""


def get_config_base_dir_from_txt(txt_path) -> str:
    with open(txt_path, "r") as f:
        for line in f:
            if not line.strip().isdigit():
                return line.strip()
    return ""


def get_ids_from_tst_config_list(txt_path: str = "", tst_config_list: list | None = None) -> set[int]:
    # If .txt provided, read tst_config_list from it
    tst_config_list = tst_config_list if tst_config_list else []
    if txt_path:
        with open(txt_path, "r") as file:
            for line in file:
                line = line.strip()
                tst_config_list.append(line)

    task_ids = set()
    for tst_config in tst_config_list:
        # Case 1: JSON file path pattern (e.g., "config_files/vwa/test_shopping/102.json")
        if tst_config.endswith(".json"):
            match = re.search(r"(\d+)\.json$", tst_config)
            if match:
                task_ids.add(int(match.group(1)))
            continue

        # Case 2: Plain integer (e.g., "102")
        try:
            task_ids.add(int(tst_config))
        except ValueError:
            continue

    return task_ids


def get_task_ids_from_dir(dir_path: str) -> set[int]:
    dir_path = re.sub(r"-raw", "", dir_path)
    task_ids = set()
    for file in os.listdir(dir_path):
        task_id = get_task_id_from_file_path(file)
        if task_id:
            task_ids.add(int(task_id))
    return task_ids


def get_task_ids_from_csv(csv_path: str | Path) -> set[int]:
    try:
        if not os.path.exists(csv_path):
            return set()
        df = pd.read_csv(csv_path)
        return set(df["task_id"].astype(int))
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return set()


def get_attribute_from_args_file(args_file_path: str, attribute: str):
    with open(args_file_path, "r") as f:
        args = json.load(f)
        return args.get(attribute, "")


def get_domain_from_args_file(args_file_path: str):
    domain = get_attribute_from_args_file(args_file_path=args_file_path, attribute="test_config_base_dir")
    if not domain:
        raise ValueError(f"Domain not found in {args_file_path}")
    return domain.split("/")[-1].split("_")[-1]


def is_experiments_dir(dir_path: str, identifier_files: list[str] = ["args.json", "log_files.txt"], match_any=True):
    """
    Returns `True` if `dir_path` and its nested subdirs contain any or all of the `identifier_files`.
    Args:
        dir_path (str): The path to start the traversal search for the `identifier_files`.
        identifier_files (list[str]): The list of files to check for.
        match_any (bool): If `True`, returns `True` if `dir_path` contains ANY of the `identifier_files`.
        If `False`, returns `True` if `dir_path` contains ALL of the `identifier_files`.
    """
    if not os.path.isdir(dir_path):
        return False

    if isinstance(identifier_files, str):
        identifier_files = [identifier_files]

    if match_any:
        return any(contains_file(dir_path, file) for file in identifier_files)
    else:
        return all(contains_file(dir_path, file) for file in identifier_files)


if __name__ == "__main__":
    pass
    # remove_empty_dirs(
    #     "results/google",
    #     exclude_dirs=[
    #         "results/google/gemini-2.0-flash-001/shopping_2025-02-16-2011/tasks2",
    #         # "google",
    #     ],
    # )

    # dir_1 = "experiments copy/____EXEC____mem=text-u-a_t=100/shopping/runs/tasks_1"
    # dir_2 = "experiments/____EXEC____mem=text-u-a_t=100/shopping/runs/tasks_1"
    # print(identical_dir_content(dir_1, dir_2, ignore_dir_struct=True))

    # copy_move(
    #     dir_1,
    #     dir_2,
    #     mode="move",
    #     overwrite_on_file_copy=False,
    #     merge_dirs=False,
    #     copy_move_only_new=True,
    # )
    # )
