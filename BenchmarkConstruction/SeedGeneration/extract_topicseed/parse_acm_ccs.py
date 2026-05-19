from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

SKOS_NS = "http://www.w3.org/2004/02/skos/core#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
XML_NS = "http://www.w3.org/XML/1998/namespace"

NS = {
    "skos": SKOS_NS,
    "rdf": RDF_NS,
    "xml": XML_NS,
}

DEFAULT_INPUT = Path("extract_topicseed/acm_ccs2012-1626988337597.xml")
DEFAULT_TREE_OUTPUT = Path("extract_topicseed/acm_ccs2012_tree.json")
DEFAULT_FLAT_OUTPUT = Path("extract_topicseed/acm_ccs2012_flat.jsonl")
DEFAULT_LEAF_OUTPUT = Path("extract_topicseed/acm_ccs2012_leaf_nodes.jsonl")
DEFAULT_LEAF_LABELS_OUTPUT = Path("extract_topicseed/acm_ccs2012_leaf_labels.txt")


@dataclass(slots=True)
class ConceptRecord:
    concept_id: str
    label: str
    broader_id: str | None
    narrower_ids: list[str]
    is_top_concept: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse an ACM CCS RDF/SKOS XML taxonomy into JSON outputs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the ACM CCS XML file.",
    )
    parser.add_argument(
        "--tree-output",
        type=Path,
        default=DEFAULT_TREE_OUTPUT,
        help="Path to the hierarchical JSON output.",
    )
    parser.add_argument(
        "--flat-output",
        type=Path,
        default=DEFAULT_FLAT_OUTPUT,
        help="Path to the flat JSONL output.",
    )
    parser.add_argument(
        "--leaf-output",
        type=Path,
        default=DEFAULT_LEAF_OUTPUT,
        help="Path to the leaf-only JSONL output.",
    )
    parser.add_argument(
        "--leaf-labels-output",
        type=Path,
        default=DEFAULT_LEAF_LABELS_OUTPUT,
        help="Path to the comma-separated leaf labels output.",
    )
    return parser.parse_args()


def extract_resource_id(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    resource = element.attrib.get(f"{{{RDF_NS}}}resource")
    return resource or None


def extract_label(concept: ET.Element) -> str:
    pref_labels = concept.findall("skos:prefLabel", NS)
    english_label: str | None = None
    fallback_label: str | None = None

    for label in pref_labels:
        text = (label.text or "").strip()
        if not text:
            continue
        if fallback_label is None:
            fallback_label = text

        language = label.attrib.get("lang") or label.attrib.get(f"{{{XML_NS}}}lang")
        if language == "en":
            english_label = text
            break

    if english_label is not None:
        return english_label
    if fallback_label is not None:
        return fallback_label

    concept_id = concept.attrib.get(f"{{{RDF_NS}}}about", "<unknown>")
    raise ValueError(f"Missing prefLabel for concept {concept_id}")


def parse_xml(xml_path: Path) -> tuple[str | None, list[str], dict[str, ConceptRecord]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    scheme = root.find("skos:ConceptScheme", NS)
    scheme_id = None
    top_concept_ids: list[str] = []
    if scheme is not None:
        scheme_id = scheme.attrib.get(f"{{{RDF_NS}}}about")
        top_concept_ids = [
            resource_id
            for top_concept in scheme.findall("skos:hasTopConcept", NS)
            if (resource_id := extract_resource_id(top_concept)) is not None
        ]

    concepts: dict[str, ConceptRecord] = {}
    for concept in root.findall("skos:Concept", NS):
        concept_id = concept.attrib.get(f"{{{RDF_NS}}}about")
        if concept_id is None:
            raise ValueError("Encountered skos:Concept without rdf:about")

        broader_id = extract_resource_id(concept.find("skos:broader", NS))
        narrower_ids = [
            resource_id
            for child in concept.findall("skos:narrower", NS)
            if (resource_id := extract_resource_id(child)) is not None
        ]
        concepts[concept_id] = ConceptRecord(
            concept_id=concept_id,
            label=extract_label(concept),
            broader_id=broader_id,
            narrower_ids=narrower_ids,
            is_top_concept=concept_id in top_concept_ids,
        )

    for concept in concepts.values():
        if concept.broader_id is None:
            continue
        parent = concepts.get(concept.broader_id)
        if parent is None:
            raise ValueError(
                f"Concept {concept.concept_id} references missing parent {concept.broader_id}"
            )
        if concept.concept_id not in parent.narrower_ids:
            parent.narrower_ids.append(concept.concept_id)

    for concept in concepts.values():
        concept.narrower_ids.sort()

    return scheme_id, top_concept_ids, concepts


def build_tree_node(
    concept_id: str,
    concepts: dict[str, ConceptRecord],
    depth: int,
) -> dict[str, Any]:
    concept = concepts[concept_id]
    child_nodes = [
        build_tree_node(child_id, concepts, depth + 1) for child_id in concept.narrower_ids
    ]
    return {
        "id": concept.concept_id,
        "label": concept.label,
        "is_top_concept": concept.is_top_concept,
        "is_leaf": len(child_nodes) == 0,
        "depth": depth,
        "children": child_nodes,
    }


def build_flat_records(
    top_concept_ids: list[str],
    concepts: dict[str, ConceptRecord],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def visit(concept_id: str, depth: int, path: list[str]) -> None:
        concept = concepts[concept_id]
        current_path = [*path, concept.label]
        records.append(
            {
                "id": concept.concept_id,
                "label": concept.label,
                "parent_id": concept.broader_id,
                "child_ids": concept.narrower_ids,
                "is_top_concept": concept.is_top_concept,
                "is_leaf": len(concept.narrower_ids) == 0,
                "depth": depth,
                "path_labels": current_path,
            }
        )
        for child_id in concept.narrower_ids:
            visit(child_id, depth + 1, current_path)

    for top_concept_id in top_concept_ids:
        visit(top_concept_id, 0, [])

    return records


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_tree_json(
    output_path: Path,
    scheme_id: str | None,
    top_concept_ids: list[str],
    concepts: dict[str, ConceptRecord],
) -> dict[str, int]:
    top_concepts = [build_tree_node(concept_id, concepts, 0) for concept_id in top_concept_ids]
    leaf_count = sum(1 for concept in concepts.values() if not concept.narrower_ids)
    max_depth = max(
        (len(concept.concept_id.split(".")) - 1 for concept in concepts.values()),
        default=0,
    )
    stats = {
        "total_nodes": len(concepts),
        "top_concepts": len(top_concept_ids),
        "leaf_nodes": leaf_count,
        "max_depth": max_depth,
    }

    payload = {
        "scheme_id": scheme_id,
        "stats": stats,
        "top_concepts": top_concepts,
    }

    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    return stats


def write_flat_jsonl(
    output_path: Path,
    top_concept_ids: list[str],
    concepts: dict[str, ConceptRecord],
) -> None:
    records = build_flat_records(top_concept_ids, concepts)
    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")


def write_leaf_jsonl(
    output_path: Path,
    top_concept_ids: list[str],
    concepts: dict[str, ConceptRecord],
) -> None:
    ensure_parent(output_path)

    with output_path.open("w", encoding="utf-8") as handle:
        def visit(concept_id: str, depth: int) -> None:
            concept = concepts[concept_id]
            if not concept.narrower_ids:
                handle.write(
                    json.dumps(
                        {
                            "id": concept.concept_id,
                            "label": concept.label,
                            "depth": depth,
                        },
                        ensure_ascii=True,
                    )
                )
                handle.write("\n")
                return

            for child_id in concept.narrower_ids:
                visit(child_id, depth + 1)

        for top_concept_id in top_concept_ids:
            visit(top_concept_id, 0)


def write_leaf_labels(
    output_path: Path,
    top_concept_ids: list[str],
    concepts: dict[str, ConceptRecord],
) -> None:
    labels: list[str] = []

    def visit(concept_id: str) -> None:
        concept = concepts[concept_id]
        if not concept.narrower_ids:
            labels.append(concept.label)
            return

        for child_id in concept.narrower_ids:
            visit(child_id)

    for top_concept_id in top_concept_ids:
        visit(top_concept_id)

    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(labels))
        handle.write("\n")


def main() -> None:
    args = parse_args()
    scheme_id, top_concept_ids, concepts = parse_xml(args.input)
    stats = write_tree_json(args.tree_output, scheme_id, top_concept_ids, concepts)
    write_flat_jsonl(args.flat_output, top_concept_ids, concepts)
    write_leaf_jsonl(args.leaf_output, top_concept_ids, concepts)
    write_leaf_labels(args.leaf_labels_output, top_concept_ids, concepts)

    print(f"Parsed scheme: {scheme_id or 'unknown'}")
    print(f"Total nodes: {stats['total_nodes']}")
    print(f"Top concepts: {stats['top_concepts']}")
    print(f"Leaf nodes: {stats['leaf_nodes']}")
    print(f"Max depth: {stats['max_depth']}")
    print(f"Tree JSON written to: {args.tree_output}")
    print(f"Flat JSONL written to: {args.flat_output}")
    print(f"Leaf JSONL written to: {args.leaf_output}")
    print(f"Leaf labels written to: {args.leaf_labels_output}")


if __name__ == "__main__":
    main()
