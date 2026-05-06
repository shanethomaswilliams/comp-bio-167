import pandas as pd
import argparse
import json
import os
from collections import defaultdict

def parse_go_obo(obo_path):
    print(f"Parsing OBO file from: {obo_path}")
    go_namespace = {}
    parents = defaultdict(set)
    
    current_id = None
    current_namespace = None
    current_parents = []

    with open(obo_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if line == "[Term]":
                if current_id is not None:
                    go_namespace[current_id] = current_namespace
                    for p in current_parents:
                        parents[current_id].add(p)

                current_id = None
                current_namespace = None
                current_parents = []
                continue

            if line.startswith("id: GO:"):
                current_id = line.replace("id: ", "").strip()
            elif line.startswith("namespace: "):
                ns = line.replace("namespace: ", "").strip()
                if ns == "biological_process":
                    current_namespace = "BPO"
                elif ns == "cellular_component":
                    current_namespace = "CCO"
                elif ns == "molecular_function":
                    current_namespace = "MFO"
            elif line.startswith("is_a: GO:"):
                parent_id = line.split()[1]
                current_parents.append(parent_id)

        if current_id is not None:
            go_namespace[current_id] = current_namespace
            for p in current_parents:
                parents[current_id].add(p)

    return go_namespace, parents

def build_ancestor_lookup(go_namespace, go_parents):
    print("Building ancestor lookup tree...")
    ancestor_lookup = {}

    for term in go_namespace.keys():
        visited = set()
        stack = list(go_parents.get(term, []))

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(go_parents.get(node, []))

        ancestor_lookup[term] = visited

    return ancestor_lookup

def propagate_scores_upward(df, ancestor_lookup):
    protein_to_scores = defaultdict(dict)

    for row in df.itertuples(index=False):
        protein = row.protein_id
        term = row.go_term
        score = float(row.confidence)

        current = protein_to_scores[protein].get(term, 0.0)
        if score > current:
            protein_to_scores[protein][term] = score

        for anc in ancestor_lookup.get(term, set()):
            current_anc = protein_to_scores[protein].get(anc, 0.0)
            if score > current_anc:
                protein_to_scores[protein][anc] = score

    rows = []
    for protein, score_map in protein_to_scores.items():
        for term, score in score_map.items():
            rows.append((protein, term, score))

    return pd.DataFrame(rows, columns=["protein_id", "go_term", "confidence"])

def process_model(model_name, filepath, ancestor_lookup, output_dir):
    print(f"\n--- Processing: {model_name} ---")
    print(f"Loading predictions from: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found -> {filepath}. Skipping.")
        return

    df = pd.read_csv(filepath, sep="\t", header=None, names=["protein_id", "go_term", "confidence"])
    
    # Safety cleaning
    df["protein_id"] = df["protein_id"].astype(str).str.strip()
    df["go_term"] = df["go_term"].astype(str).str.strip()
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(0.0)

    # Propagate
    print(f"Propagating {len(df)} rows...")
    propagated_df = propagate_scores_upward(df, ancestor_lookup)

    # Sort for standard formatting
    propagated_df = propagated_df.sort_values(["protein_id", "go_term"]).reset_index(drop=True)
    
    # Save output
    out_path = os.path.join(output_dir, f"{model_name}_propagated.tsv")
    print(f"Saving to: {out_path}")
    propagated_df.to_csv(out_path, sep="\t", index=False, header=False)


def main():
    parser = argparse.ArgumentParser(description="Batch apply GO-DAG propagation using a JSON config.")
    parser.add_argument("--config", required=True, help="Path to the JSON configuration file")
    args = parser.parse_args()

    # Load JSON
    with open(args.config, "r") as f:
        config = json.load(f)

    obo_path = config.get("go_obo")
    output_dir = config.get("output_dir", "./")
    models = config.get("models", {})

    os.makedirs(output_dir, exist_ok=True)

    # 1. Parse GO structure ONCE
    print("=== Initializing GO-DAG ===")
    go_namespace, go_parents = parse_go_obo(obo_path)
    ancestor_lookup = build_ancestor_lookup(go_namespace, go_parents)

    # 2. Iterate through all models in the JSON
    for model_name, filepath in models.items():
        process_model(model_name, filepath, ancestor_lookup, output_dir)

    print("\n=== Batch Processing Complete! ===")

if __name__ == "__main__":
    main()