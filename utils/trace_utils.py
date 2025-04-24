import os
import re

import pandas as pd
from bs4 import BeautifulSoup, Comment

from utils.file_utils import resolve_path_conflict
from utils.image_utils import any_to_b64
from vwa_utils.extract_trajectory_html import parse_webpage_states, process_html_file, rebuild_trajectory_vwa_format

# TODO: confirmation or redundancy prevention if annotation_banner already exists
# TODO: support list of [text,image] in build_annotation_banner. Use `is_image` from utils
# =======================================================================================================================
# LINK Helpers
# =======================================================================================================================


# I/O helpers
def write_file(file_path: str, content: str, overwrite: bool = False, out_file: str = "") -> str:
    if overwrite:
        out_file = file_path
    else:
        out_file = out_file or str(resolve_path_conflict(file_path))

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(content)

    return out_file


# HTML Helpers
def build_annotation_banner(
    annotations: list[str] | str,
    pin: bool = False,
    color: str = "lightgray",
    font_size: str = "1em",
) -> str:
    """
    Creates an annotation banner for an HTML document.

    If append is False, returns a complete <div> element with the provided annotations,
    each separated by a <br> tag.
    If append is True, returns only the annotation snippet to be appended to an existing banner.

    Parameters:
        annotations: A string or list of annotation strings.
        pin: Whether to apply sticky positioning.
        color: Background color.
        font_size: Font size for the annotations.
        append: If True, returns a snippet for appending to an existing banner.

    Returns:
        A string of HTML representing the annotation(s).
    """
    if isinstance(annotations, str):
        annotations = [annotations]
    # Join annotations separated by <br>
    ann_html = "<br>".join(ann.replace("\n", "<br>") for ann in annotations)

    if pin:
        style = f"position:sticky;top:0;left:0;width:100%;background:{color};padding:5px;z-index:9999;text-align:left;font-size:{font_size};"
    else:
        style = f"background:{color};padding:5px;text-align:left;font-size:{font_size};"
    return f"<div id='annotation_banner' style='{style}'>{ann_html}</div>\n"


def insert_top_block(content: str, block: str) -> str:
    """
    Insert the provided HTML block immediately after the opening <body> tag.
    Falls back to using the <head> tag or prepending if neither is found.
    """
    body_match = re.search(r"(<body[^>]*>)", content, flags=re.IGNORECASE | re.DOTALL)
    if body_match:
        insertion_point = body_match.end()
        new_content = content[:insertion_point] + "\n" + block + content[insertion_point:]
    else:
        head_match = re.search(r"(<head[^>]*>)", content, flags=re.IGNORECASE | re.DOTALL)
        if head_match:
            insertion_point = head_match.end()
            new_content = content[:insertion_point] + "\n" + block + content[insertion_point:]
        else:
            new_content = block + content
    return new_content


# =======================================================================================================================
# LINK Annotators
# =======================================================================================================================


def annotate_html(
    html_path: str,
    annotations: list[str] | str,
    pin_annotations: list[bool] | bool = True,
    annotation_color: str = "lightgray",
    write_to_file: bool = False,
    overwrite: bool = False,
    out_file: str = "",
    font_size: str = "1em",
) -> tuple[str, str]:
    """
    Annotate an HTML file by inserting an annotation banner immediately after the opening
    <body> (or <head> as a fallback). Some of the annotations will be "pinned" (sticky)
    so that they remain visible when scrolling.

    Parameters:
        html_path: Path to the HTML file.
        annotations: A string or list of annotation strings.
        pin_annotations: A boolean or list of booleans indicating whether each annotation is pinned.
        annotation_color: Background color for the annotations.
        write_to_file: If True, writes the new HTML content back to a file.
        overwrite: Whether to overwrite the original file.
        out_file: The file path to write the new content.
        font_size: The font size for the annotations.

    Returns:
        A tuple of the new HTML content and the output file path (if written) or an empty string.
    """
    # Convert single annotation or boolean value to list.
    if isinstance(annotations, str):
        annotations = [annotations]
    if isinstance(pin_annotations, bool):
        pin_annotations = [pin_annotations] * len(annotations)

    # Read the HTML content.
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Separate pinned and non-pinned annotations.
    pinned_annotations = [ann for ann, pin in zip(annotations, pin_annotations) if pin]
    non_pinned_annotations = [ann for ann, pin in zip(annotations, pin_annotations) if not pin]

    # Build HTML snippets for the annotation banners.
    pinned_html = ""
    non_pinned_html = ""

    if pinned_annotations:
        pinned_html = build_annotation_banner(pinned_annotations, pin=True, color=annotation_color, font_size=font_size)
    if non_pinned_annotations:
        non_pinned_html = build_annotation_banner(
            non_pinned_annotations, pin=False, color=annotation_color, font_size=font_size
        )
        # Remove the id attribute to avoid duplicates.
        non_pinned_html = non_pinned_html.replace("id='annotation_banner'", "")

    # Combine the banners so that the pinned ones come first.
    banner_html = pinned_html + non_pinned_html

    # Insert the banner block after the opening <body> (or <head> as a fallback).
    new_content = insert_top_block(content, banner_html)

    if write_to_file:
        written_file = write_file(html_path, new_content, overwrite=overwrite, out_file=out_file)
        return new_content, written_file
    else:
        return new_content, ""


def annotate_txt_file(
    txt_path: str,
    annotation: str,
    write_to_file: bool = False,
    overwrite: bool = False,
    out_file: str = "",
) -> tuple[str, str]:
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = annotation + "\n" + content

    if write_to_file:
        return new_content, write_file(txt_path, new_content, overwrite=overwrite, out_file=out_file)
    else:
        return new_content, ""


# =======================================================================================================================
# LINK VWA-Specific
# =======================================================================================================================


def build_intent_info(
    content: str,
    data: dict,
    add_intent_images: bool = True,
    highlight_intent: bool = False,
    img_size: str = "10%",
    font_size: str = ".8em",
) -> str:
    """
    Extract the first occurrence of 'intent:' from the first <pre> block and any intent images.
    If highlight_intent is True, the intent text is wrapped in a <span> with a light coral background.

    Returns:
        A string of HTML that includes the intent text (highlighted if requested) and any intent images (converted to base64),
        arranged side-by-side if there is more than one image.
    """
    intent_text = ""
    # Extract the first <pre> block.
    pre_match = re.search(r"<pre[^>]*>(.*?)</pre>", content, flags=re.IGNORECASE | re.DOTALL)
    if pre_match:
        pre_content = pre_match.group(1)
        for line in pre_content.splitlines():
            if "intent:" in line:
                intent_text = line.strip()
                break

    html_parts = []
    if intent_text:
        if highlight_intent:
            # Highlight the text after 'intent:' within the banner and make it larger
            intent_text = re.sub(
                r"(intent:\s*)(.*)",
                r"\1<span style='background: lightcoral; font-size: {font_size};'>\2</span>",
                intent_text,
            )
        html_parts.append(intent_text)

    if add_intent_images and data.get("intent_images"):
        images_tags = []
        for img in data["intent_images"]:
            try:
                img_b64 = any_to_b64(img)
                # Remove display:block and use inline-block to allow side by side.
                img_tag = f"<img src='{img_b64}' style='max-width:{img_size}; display:inline-block;'>"
            except Exception as e:
                img_tag = f"<!-- Could not process image {img}; error: {e} -->"
            images_tags.append(img_tag)
        if images_tags:
            # Wrap multiple images in a flex container, so they display side by side.
            images_container = (
                "<div style='display: flex; flex-wrap: wrap; gap: 10px; margin-top: 10px;'>"
                + "".join(images_tags)
                + "</div>"
            )
            html_parts.append(images_container)

    return "<br>".join(html_parts)


def annotate_html_vwa(
    html_path: str,
    annotations: list[str] | str = "",
    pin_annotations: list[bool] | bool = True,
    highlight_intent: bool = True,
    add_intent_images: bool = True,
    write_to_file: bool = False,
    overwrite: bool = False,
    out_file: str = "",
    img_size: str = "5%",
    font_size: str = ".8em",
) -> tuple[str, str]:
    # Read the original HTML content.
    with open(html_path, "r", encoding="utf-8") as f:
        original_content = f.read()

    # Regularize parameters for annotations.
    if isinstance(annotations, str):
        annotations = [annotations] if annotations else []

    # Regularize pin_annotations.
    if isinstance(pin_annotations, bool):
        pin_ann_list = [pin_annotations] * len(annotations)
    else:
        pin_ann_list = list(pin_annotations)
        # Ensure length matches annotations list.
        if len(pin_ann_list) != len(annotations):
            print(
                f"Warning: pin_annotations length {len(pin_ann_list)} does not match annotations length {len(annotations)}. Using first value for all."
            )
            pin_ann_list = [pin_ann_list[0]] * len(annotations)

    # Extract supplementary data (including intent images) from the HTML.
    data = parse_webpage_states(original_content, stop_at_critique=False)

    # Build the intent information.
    intent_info = build_intent_info(
        original_content,
        data,
        add_intent_images=add_intent_images,
        highlight_intent=highlight_intent,
        img_size=img_size,
    )

    # Combine the intent info with the provided annotations.
    combined_annotations = []
    combined_pin_annotations = []
    if intent_info:
        combined_annotations.append(intent_info)
        combined_pin_annotations.append(True)
    if annotations:
        combined_annotations.extend(annotations)
        combined_pin_annotations.extend(pin_ann_list)

    # If there's nothing to annotate, return empty.
    if not combined_annotations:
        return "", ""

    # Apply the annotation to the HTML file.
    new_content, out_file = annotate_html(
        html_path,
        combined_annotations,
        pin_annotations=combined_pin_annotations,
        write_to_file=write_to_file,
        overwrite=overwrite,
        out_file=out_file,
        font_size=font_size,
    )
    return new_content, out_file


def annotate_txt_vwa(
    txt_path: str,
    annotations: list[str] | str = "",
    overwrite: bool = False,
    out_file: str = "",
    write_to_file: bool = False,
) -> tuple[str, str]:
    if isinstance(annotations, list):
        annotations = "\n".join(annotations)
    return annotate_txt_file(txt_path, annotations, write_to_file=write_to_file, overwrite=overwrite, out_file=out_file)


def annotate_csv_vwa(
    csv_path: str,
    new_fields: dict = {},
    overwrite: bool = False,
    out_file: str = "",
    write_to_file: bool = False,
) -> tuple[str, str]:
    # Read the CSV into a DataFrame.
    df = pd.read_csv(csv_path)

    # For each new field, add a new column with "ann:" prepended to avoid modifying existing fields.
    for key, value in new_fields.items():
        base_col = "ann:" + key
        new_col = base_col
        count = 1
        # Ensure the new column name is unique.
        while new_col in df.columns:
            new_col = f"{base_col}_{count}"
            count += 1

        # Add the new column with empty strings.
        df[new_col] = ""

        # Assign the provided value to the first row.
        if len(df) > 0:
            df.at[df.index[0], new_col] = value
        else:
            # In case df is completely empty, create a new row.
            df = pd.DataFrame({new_col: [value]})

    # Convert DataFrame back to CSV string.
    csv_content = df.to_csv(index=False)

    if write_to_file:
        if overwrite:
            out_file = csv_path
        else:
            out_file = out_file or str(resolve_path_conflict(csv_path))
        df.to_csv(out_file, index=False)
        return csv_content, out_file
    else:
        return csv_content, ""


def annotate_vwa_trace(
    trace_path: str,
    annotations: list[str] | str = "",
    pin_annotations: list[bool] | bool = True,
    overwrite: bool = False,
    out_file: str = "",
    write_to_file: bool = False,
) -> tuple[str, str]:
    # Regularize params
    if isinstance(annotations, str):
        annotations = [annotations]

    if isinstance(pin_annotations, bool):
        pin_annotations = [pin_annotations] * len(annotations)

    # HTML trace
    if trace_path.endswith(".html"):
        return annotate_html_vwa(
            html_path=trace_path,
            annotations=annotations,
            pin_annotations=pin_annotations,
            write_to_file=write_to_file,
            overwrite=overwrite,
            out_file=out_file,
        )
    # TXT trace
    elif trace_path.endswith(".txt"):
        return annotate_txt_vwa(
            trace_path, annotations, write_to_file=write_to_file, overwrite=overwrite, out_file=out_file
        )

    # CSV trace
    elif trace_path.endswith(".csv"):
        new_fields = {}
        for annotation in annotations:
            try:
                # Handle both single-line and multi-line annotations
                if "\n" in annotation:
                    fields = {
                        line.split(":")[0].strip(): line.split(":", 1)[1].strip()
                        for line in annotation.split("\n")
                        if ":" in line
                    }
                else:
                    # Handle single line case
                    if ":" in annotation:
                        key = annotation.split(":")[0].strip()
                        value = annotation.split(":", 1)[1].strip()
                        fields = {key: value}
                    else:
                        # If no colon is found, use the entire annotation as a default field
                        fields = {"annotation": annotation.strip()}
                new_fields.update(fields)
            except Exception as e:
                print(f"Warning: error parsing {annotation}: {e} for csv.")
                continue
        if not new_fields:
            return "", ""

        return annotate_csv_vwa(
            trace_path, new_fields, write_to_file=write_to_file, overwrite=overwrite, out_file=out_file
        )
    else:
        return "", ""


if __name__ == "__main__":
    # Example usage:
    html_path = "render_186.html"
    content, out_file = annotate_html_vwa(
        html_path,
        annotations=["A", "B", "C"],
        pin_annotations=[False, True, False],
        highlight_intent=True,
        add_intent_images=True,
        write_to_file=True,
        overwrite=False,
        out_file="0b.html",
        img_size="8%",
        font_size=".8em",
    )

    trajectory, meta_data, trajectory_view = rebuild_trajectory_vwa_format(process_html_file(out_file))
