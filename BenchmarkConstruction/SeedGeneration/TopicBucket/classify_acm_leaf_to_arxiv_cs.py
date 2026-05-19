from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from paperbench.io_utils import append_jsonl  # noqa: E402
from paperbench.llm.client import OpenAICompatibleClient  # noqa: E402

DEFAULT_TREE_PATH = Path(__file__).with_name("acm_ccs2012_tree.json")
DEFAULT_TAXONOMY_PATH = Path(__file__).with_name("cs_taxonomy.jsonl")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("acm_ccs2012_leaf_to_arxiv_cs.jsonl")
DEFAULT_CACHE_PATH = Path(__file__).with_name("acm_ccs2012_leaf_to_arxiv_cs_cache.jsonl")
DEFAULT_API_KEY = "sk-eq8RqEUVNCjUyhE5qLdBwtAEmeQQsYDzVBpqmsZLQzZFWDtp"
DEFAULT_BASE_URL = "https://api2.aigcbest.top/v1"
DEFAULT_MODEL = "gpt-5.3-chat-latest"


@dataclass(slots=True)
class ArxivCategory:
    code: str
    name: str
    description: str


@dataclass(slots=True)
class AcmLeafNode:
    acm_id: str
    leaf_label: str
    depth: int
    path_labels: list[str]

    @property
    def path_text(self) -> str:
        return " -> ".join(self.path_labels)


class ClassificationResponse(BaseModel):
    primary_category: Optional[str] = None
    matched_categories: list[str] = Field(default_factory=list)
    discarded: bool
    reason: str = ""


class OutputRecord(BaseModel):
    acm_id: str
    leaf_label: str
    depth: int
    path_labels: list[str]
    path_text: str
    primary_category: Optional[str] = None
    matched_categories: list[str]
    discarded: bool
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify ACM CCS leaf nodes into one or more arXiv cs categories."
    )
    parser.add_argument(
        "--tree",
        type=Path,
        default=DEFAULT_TREE_PATH,
        help="Path to the ACM CCS tree JSON file.",
    )
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=DEFAULT_TAXONOMY_PATH,
        help="Path to the arXiv cs taxonomy JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the output JSONL file.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help="Path to the LLM response cache JSONL file.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name for classification. Defaults to gpt-5.3-chat-latest.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of pending leaf nodes to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file instead of resuming from existing records.",
    )
    return parser.parse_args()


def load_tree(tree_path: Path) -> dict[str, Any]:
    with tree_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {tree_path}")
    return payload


def extract_leaf_nodes(tree_payload: dict[str, Any]) -> list[AcmLeafNode]:
    top_concepts = tree_payload.get("top_concepts")
    if not isinstance(top_concepts, list):
        raise ValueError("Tree payload is missing a valid 'top_concepts' list.")

    leaves: list[AcmLeafNode] = []

    def visit(node: dict[str, Any], path_labels: list[str]) -> None:
        label = node.get("label")
        node_id = node.get("id")
        depth = node.get("depth")
        is_leaf = node.get("is_leaf")
        children = node.get("children")

        if not isinstance(label, str) or not isinstance(node_id, str):
            raise ValueError("Encountered node without valid 'id' or 'label'.")
        if not isinstance(depth, int) or not isinstance(is_leaf, bool):
            raise ValueError(f"Encountered malformed node metadata for {node_id}.")
        if not isinstance(children, list):
            raise ValueError(f"Encountered node without valid children list: {node_id}")

        current_path = [*path_labels, label]
        if is_leaf:
            leaves.append(
                AcmLeafNode(
                    acm_id=node_id,
                    leaf_label=label,
                    depth=depth,
                    path_labels=current_path,
                )
            )
            return

        for child in children:
            if not isinstance(child, dict):
                raise ValueError(f"Encountered non-object child under {node_id}")
            visit(child, current_path)

    for top_concept in top_concepts:
        if not isinstance(top_concept, dict):
            raise ValueError("Encountered non-object top concept.")
        visit(top_concept, [])

    return leaves


def load_arxiv_categories(taxonomy_path: Path) -> list[ArxivCategory]:
    categories: list[ArxivCategory] = []
    with taxonomy_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on line {line_number} in {taxonomy_path}")
            code = payload.get("code")
            name = payload.get("name")
            description = payload.get("description")
            if not isinstance(code, str) or not isinstance(name, str) or not isinstance(description, str):
                raise ValueError(
                    f"Malformed category record on line {line_number} in {taxonomy_path}"
                )
            categories.append(
                ArxivCategory(code=code, name=name, description=description)
            )
    if not categories:
        raise ValueError(f"No categories loaded from {taxonomy_path}")
    return categories


def build_taxonomy_block(categories: list[ArxivCategory]) -> str:
    lines = []
    for category in categories:
        lines.append(f"{category.code} | {category.name} | {category.description}")
    return "\n".join(lines)


def build_system_prompt() -> str:
    return (
        "You are an expert taxonomy mapper.\n"
        "Your task is to map one ACM CCS leaf topic into arXiv Computer Science (cs.*) categories.\n"
        "Return strict JSON only, with no extra text.\n"
        "\n"
        "Mapping rules:\n"
        "1. Use both the leaf label and the full ACM path to determine meaning.\n"
        "2. Only map to official arXiv Computer Science category codes (cs.*).\n"
        "3. Do not output non-cs categories such as stat.*, math.*, eess.*, q-fin.*, physics.*, etc.\n"
        "4. Assign categories only when the semantic fit is direct and strong.\n"
        "6. Identify one primary category when possible.\n"
        "7. Secondary categories are optional and should only be included when clearly justified.\n"
        "8. Return at most 3 matched categories in total.\n"
        "9. If no cs category is clearly appropriate, set discarded=true, primary_category=null, and matched_categories=[].\n"
        "10. Match the core research topic, not peripheral applications or vague associations.\n"
        "\n"
        "Discard when:\n"
        "- the topic is mainly non-CS;\n"
        "- the connection to cs.* is weak or indirect.\n"
        "\n"
        "Output schema exactly:\n"
        '{"primary_category": "cs.AI", "matched_categories": ["cs.AI"], "discarded": false, "reason": "short explanation"}'
    )


def build_user_prompt(leaf: AcmLeafNode, taxonomy_block: str) -> str:
    return (
        "Classify this ACM CCS leaf node into arXiv Computer Science categories.\n\n"
        f"Leaf label: {leaf.leaf_label}\n"
        f"ACM path: {leaf.path_text}\n"
        f"ACM depth: {leaf.depth}\n\n"
        "Candidate arXiv cs categories:\n"
        f"{taxonomy_block}\n\n"
        "Return strict JSON only."
    )


def load_processed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    processed_ids: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on line {line_number} in {output_path}")
            acm_id = payload.get("acm_id")
            if not isinstance(acm_id, str):
                raise ValueError(f"Missing acm_id on line {line_number} in {output_path}")
            processed_ids.add(acm_id)
    return processed_ids


def build_client(model: str, cache_path: Path) -> OpenAICompatibleClient:
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("TOPIC_BUCKET_OPENAI_API_KEY", DEFAULT_API_KEY).strip()
    base_url = os.getenv("TOPIC_BUCKET_OPENAI_BASE_URL", DEFAULT_BASE_URL).strip()

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing.")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL is missing.")

    return OpenAICompatibleClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        cache_path=cache_path,
        temperature=0.0,
    )


def normalize_response(
    response: ClassificationResponse,
    allowed_codes: set[str],
) -> ClassificationResponse:
    filtered_codes = sorted({code for code in response.matched_categories if code in allowed_codes})[:3]
    primary_category = response.primary_category if response.primary_category in allowed_codes else None
    if primary_category is not None and primary_category not in filtered_codes:
        filtered_codes = [primary_category, *[code for code in filtered_codes if code != primary_category]][:3]
    discarded = response.discarded or not filtered_codes
    return ClassificationResponse(
        primary_category=None if discarded else (primary_category or filtered_codes[0]),
        matched_categories=[] if discarded else filtered_codes,
        discarded=discarded,
        reason=response.reason.strip(),
    )


def classify_leaf(
    *,
    client: OpenAICompatibleClient,
    leaf: AcmLeafNode,
    taxonomy_block: str,
    allowed_codes: set[str],
) -> ClassificationResponse:
    raw_response = client.generate_structured(
        system_prompt=build_system_prompt(),
        user_prompt=build_user_prompt(leaf, taxonomy_block),
        response_model=ClassificationResponse,
    )
    return normalize_response(raw_response, allowed_codes)


def make_output_record(leaf: AcmLeafNode, response: ClassificationResponse) -> OutputRecord:
    return OutputRecord(
        acm_id=leaf.acm_id,
        leaf_label=leaf.leaf_label,
        depth=leaf.depth,
        path_labels=leaf.path_labels,
        path_text=leaf.path_text,
        primary_category=response.primary_category,
        matched_categories=response.matched_categories,
        discarded=response.discarded,
        reason=response.reason,
    )


def record_to_payload(record: OutputRecord) -> dict[str, object]:
    return {
        "acm_id": record.acm_id,
        "leaf_label": record.leaf_label,
        "depth": record.depth,
        "path_labels": record.path_labels,
        "path_text": record.path_text,
        "primary_category": record.primary_category,
        "matched_categories": record.matched_categories,
        "discarded": record.discarded,
        "reason": record.reason,
    }


def main() -> None:
    args = parse_args()

    tree_payload = load_tree(args.tree)
    leaves = extract_leaf_nodes(tree_payload)
    categories = load_arxiv_categories(args.taxonomy)
    taxonomy_block = build_taxonomy_block(categories)
    allowed_codes = {category.code for category in categories}

    if args.overwrite and args.output.exists():
        args.output.unlink()

    processed_ids = set() if args.overwrite else load_processed_ids(args.output)
    pending_leaves = [leaf for leaf in leaves if leaf.acm_id not in processed_ids]
    if args.limit is not None:
        pending_leaves = pending_leaves[: args.limit]

    client = build_client(model=args.model, cache_path=args.cache)

    for index, leaf in enumerate(pending_leaves, start=1):
        response = classify_leaf(
            client=client,
            leaf=leaf,
            taxonomy_block=taxonomy_block,
            allowed_codes=allowed_codes,
        )
        record = make_output_record(leaf, response)
        append_jsonl(args.output, record_to_payload(record))
        print(
            f"[{index}/{len(pending_leaves)}] {leaf.leaf_label} -> "
            f"{','.join(record.matched_categories) if record.matched_categories else 'discarded'}"
        )

    print(f"Total leaf nodes in tree: {len(leaves)}")
    print(f"Already processed leaf nodes: {len(processed_ids)}")
    print(f"Processed in this run: {len(pending_leaves)}")
    print(f"Output written to: {args.output}")


if __name__ == "__main__":
    main()
