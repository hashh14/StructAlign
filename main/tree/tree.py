import json

import pandas as pd
from anytree import Node, RenderTree
from anytree.exporter import DictExporter


# =============================================================================
# TREE CONSTRUCTION
# =============================================================================

def build_source_2_tree_from_csv(csv_file):
    """
    Build an anytree hierarchy from a source_2 CSV file.

    The expected hierarchy columns are:
        Advertiser -> Brand Root -> Brand (Major) -> Brand (Minor) -> Brand (Leaf)

    Args:
        csv_file (str): Path to the source_2 CSV.

    Returns:
        Node: The root node of the constructed tree.
    """
    df = pd.read_csv(csv_file)
    hierarchy_columns = ["Advertiser", "Brand Root", "Brand (Major)", "Brand (Minor)", "Brand (Leaf)"]

    root = Node("Root")
    nodes = {"Root": root}

    for _, row in df.iterrows():
        parent_node = root
        path = "Root"

        for col in hierarchy_columns:
            if pd.notna(row[col]) and row[col] != "":
                current_value = row[col]
                current_path = f"{path}/{current_value}"

                if current_path not in nodes:
                    nodes[current_path] = Node(current_value, parent=parent_node)

                parent_node = nodes[current_path]
                path = current_path

    return root


def build_source_1_tree_from_csv(csv_file):
    """
    Build an anytree hierarchy from a source_1 CSV file.

    The expected hierarchy columns are: PARENT -> ADVERTISER -> BRAND

    Args:
        csv_file (str): Path to the source_1 CSV.

    Returns:
        Node: The root node of the constructed tree.
    """
    df = pd.read_csv(csv_file)
    hierarchy_columns = ["PARENT", "ADVERTISER", "BRAND"]

    root = Node("Root")
    nodes = {"Root": root}

    for _, row in df.iterrows():
        parent_node = root
        path = "Root"

        for col in hierarchy_columns:
            if pd.notna(row[col]) and row[col] != "":
                current_value = row[col]
                current_path = f"{path}/{current_value}"

                if current_path not in nodes:
                    nodes[current_path] = Node(current_value, parent=parent_node)

                parent_node = nodes[current_path]
                path = current_path

    return root


def remove_duplicate_nodes_in_branch(root):
    """
    Remove leaf nodes whose name duplicates their parent's name.

    Traversal starts from leaves and works upward so that chains of
    duplicates are resolved in a single pass.

    Args:
        root (Node): The root of the tree to deduplicate.

    Returns:
        Node: The deduplicated tree (modified in-place).
    """
    nodes_to_process = list(root.leaves)

    while nodes_to_process:
        current_batch = nodes_to_process.copy()
        nodes_to_process = []

        for node in current_batch:
            if node is root or node.parent is None:
                continue

            parent = node.parent

            if node.name == parent.name and not node.children:
                if parent not in nodes_to_process and parent is not root:
                    nodes_to_process.append(parent)
                node.parent = None
            elif parent is not root and parent not in nodes_to_process:
                nodes_to_process.append(parent)

    return root


# =============================================================================
# CONSOLE VISUALISATION
# =============================================================================

def print_tree(root):
    """Print the tree structure to stdout."""
    for pre, _, node in RenderTree(root):
        print(f"{pre}{node.name}")


def tree_to_dict(root):
    """Convert a tree to a plain dictionary structure via anytree's DictExporter."""
    return DictExporter().export(root)


def save_tree_to_text(root, output_file):
    """
    Save the rendered tree to a text file.

    Args:
        root (Node): Tree root.
        output_file (str): Destination file path.

    Returns:
        str: The output file path.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for pre, _, node in RenderTree(root):
            f.write(f"{pre}{node.name}\n")
    return output_file


# =============================================================================
# LEVEL DICTIONARY
# =============================================================================

def create_tree_level_dictionary(root):
    """
    Build a dict mapping level names ('level 0', 'level 1', …) to the
    unique node names at each depth.

    Args:
        root (Node): Tree root.

    Returns:
        dict: {level_name: [unique_name, …]}
    """
    level_dict = {}

    def process_node(node, level):
        level_key = f"level {level}"
        level_dict.setdefault(level_key, [])
        if node.name != "Root":
            level_dict[level_key].append(node.name)
        for child in node.children:
            process_node(child, level + 1)

    process_node(root, 0)

    for level in level_dict:
        level_dict[level] = list(dict.fromkeys(level_dict[level]))

    return level_dict


def save_tree_levels_to_json(level_dict, output_file):
    """
    Write a level dictionary to a JSON file.

    Args:
        level_dict (dict): Output of ``create_tree_level_dictionary``.
        output_file (str): Destination file path.

    Returns:
        str: The output file path.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(level_dict, f, indent=4)
    return output_file


# =============================================================================
# NESTED DICT / JSON SERIALISATION
# =============================================================================

def convert_tree_to_nested_dict(node):
    """
    Recursively convert an anytree Node into a nested dict.

    Each dict has the form ``{"name": str, "children": [...]}``.
    """
    return {
        "name": node.name,
        "children": [convert_tree_to_nested_dict(child) for child in node.children],
    }


def save_nested_tree_to_json(root, output_file):
    """
    Serialise a tree to a nested JSON file.

    Args:
        root (Node): Tree root.
        output_file (str): Destination file path.

    Returns:
        str: The output file path.
    """
    tree_dict = convert_tree_to_nested_dict(root)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(tree_dict, f, indent=4)
    return output_file


def load_nested_tree_from_json(json_file):
    """
    Load a nested-dict tree structure from a JSON file.

    Args:
        json_file (str): Source file path.

    Returns:
        dict: The nested tree dict.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# TREE TRAVERSAL HELPERS
# =============================================================================

def find_node_by_path(tree_dict, path):
    """
    Follow a sequence of node names from the root and return the target node.

    Args:
        tree_dict (dict): Nested tree dict.
        path (list[str]): Node names from root to target.

    Returns:
        dict | None: The matching node dict, or None if not found.
    """
    current = tree_dict
    start_idx = 1 if path[0] == "Root" and tree_dict["name"] == "Root" else 0

    for name in path[start_idx:]:
        found = False
        for child in current["children"]:
            if child["name"] == name:
                current = child
                found = True
                break
        if not found:
            return None

    return current


def get_all_nodes_at_level(tree_dict, level):
    """
    Return all nodes at a specific depth (0 = root).

    Args:
        tree_dict (dict): Nested tree dict.
        level (int): Target depth.

    Returns:
        list[dict]: Node dicts at the requested level.
    """
    if level == 0:
        return [tree_dict]

    result = []

    def collect_nodes(node, current_level):
        if current_level == level:
            result.append(node)
            return
        for child in node["children"]:
            collect_nodes(child, current_level + 1)

    collect_nodes(tree_dict, 0)
    return result


def get_all_paths(tree_dict):
    """
    Return all root-to-leaf paths as lists of node names.

    Args:
        tree_dict (dict): Nested tree dict.

    Returns:
        list[list[str]]: Each inner list is one root-to-leaf path.
    """
    paths = []

    def collect_paths(node, current_path):
        new_path = current_path + [node["name"]]
        if not node["children"]:
            paths.append(new_path)
        else:
            for child in node["children"]:
                collect_paths(child, new_path)

    collect_paths(tree_dict, [])
    return paths


def get_children_names(tree_dict, path):
    """
    Return the names of a node's direct children.

    Args:
        tree_dict (dict): Nested tree dict.
        path (list[str]): Path from root to the parent node.

    Returns:
        list[str]: Children names, or empty list if not found.
    """
    node = find_node_by_path(tree_dict, path)
    if node is None:
        return []
    return [child["name"] for child in node["children"]]


def get_parent_node(tree_dict, path):
    """
    Return the parent node and its path for a given node path.

    Args:
        tree_dict (dict): Nested tree dict.
        path (list[str]): Path from root to the target node.

    Returns:
        tuple[dict | None, list | None]: (parent node dict, parent path).
    """
    if len(path) <= 1:
        return None, None
    parent_path = path[:-1]
    parent_node = find_node_by_path(tree_dict, parent_path)
    return parent_node, parent_path


# =============================================================================
# SEARCH & SUBTREE EXTRACTION
# =============================================================================

def find_company_subtree(tree_dict, company_name):
    """
    Search the tree for a company and return its subtree and path from root.

    Args:
        tree_dict (dict): Nested tree dict.
        company_name (str): Name to search for.

    Returns:
        tuple[dict | None, list | None]:
            (company node dict, path as list of names), or (None, None).
    """
    found_paths = []
    found_nodes = []

    def search_in_tree(node, current_path):
        if node["name"] == company_name:
            found_paths.append(current_path + [node["name"]])
            found_nodes.append(node)
        for child in node["children"]:
            search_in_tree(child, current_path + [node["name"]])

    search_in_tree(tree_dict, [])

    if not found_paths:
        return None, None

    return found_nodes[0], found_paths[0]


def find_company_children(tree_dict, company_name):
    """
    Find a company node and return it with its direct children only.

    Args:
        tree_dict (dict): Nested tree dict.
        company_name (str): Name to search for.

    Returns:
        tuple[dict | None, None]: (node dict with children, None).
            Returns (None, None) if not found.
    """
    def search_in_tree(node):
        if node["name"] == company_name:
            return {"name": node["name"], "children": node["children"]}, None
        for child in node["children"]:
            result = search_in_tree(child)
            if result[0]:
                return result
        return None, None

    return search_in_tree(tree_dict)


def save_company_subtree(tree_dict, company_name, output_file=None):
    """
    Save a company's subtree to a JSON file.

    Args:
        tree_dict (dict): Nested tree dict.
        company_name (str): Company name to extract.
        output_file (str, optional): Destination path; auto-generated if omitted.

    Returns:
        dict | None: The company's subtree dict, or None if not found.
    """
    company_node, _ = find_company_subtree(tree_dict, company_name)

    if company_node is None:
        print(f"Company '{company_name}' not found in the tree structure.")
        return None

    if output_file is None:
        safe_name = "".join(c if c.isalnum() else "_" for c in company_name)
        output_file = f"{safe_name}_subtree.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(company_node, f, indent=4)

    print(f"Subtree for '{company_name}' saved to {output_file}")
    return company_node


def search_and_print_company(json_file, company_name, output_file, save_subtree=False):
    """
    Load a nested-JSON tree, find a company, and optionally save its subtree.

    Args:
        json_file (str): Path to the nested tree JSON file.
        company_name (str): Company to search for.
        output_file (str): Destination path for the subtree JSON (if saving).
        save_subtree (bool): Whether to write the subtree to disk.

    Returns:
        dict | None: The company node dict (with children), or None if not found.
    """
    tree_dict = load_nested_tree_from_json(json_file)
    company_node, _ = find_company_children(tree_dict, company_name)

    if save_subtree and company_node is not None:
        save_company_subtree(tree_dict, company_name, output_file)

    return company_node


# =============================================================================
# NAME EXTRACTION
# =============================================================================

def extract_names_from_tree(json_file):
    """
    Extract all node names from a nested-JSON tree file.

    Args:
        json_file (str): Path to the nested tree JSON file.

    Returns:
        list[str]: All names in depth-first order.
    """
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)

    names = []

    def traverse(node):
        names.append(node["name"])
        for child in node.get("children", []):
            traverse(child)

    traverse(data)
    return names


# =============================================================================
# PIPELINE ENTRY POINT
# =============================================================================

def process_trees_to_nested_json(
    source_2_csv="../source_2_top500_parents.csv",
    source_1_csv="../source_1_top1000_parents.csv",
    source_2_out="source_2_nested_tree.json",
    source_1_out="source_1_nested_tree.json",
):
    """
    Build, deduplicate, and serialise both source_2 and source_1 trees.

    Run this once to generate the nested JSON files required by the main pipeline.

    Args:
        source_2_csv (str): Path to the source_2 CSV.
        source_1_csv (str): Path to the source_1 CSV.
        source_2_out (str): Output path for the source_2 nested JSON.
        source_1_out (str): Output path for the source_1 nested JSON.
    """
    print("Building trees...")
    source_2_root = build_source_2_tree_from_csv(source_2_csv)
    source_1_root = build_source_1_tree_from_csv(source_1_csv)
    print("Tree building complete.")

    source_2_root = remove_duplicate_nodes_in_branch(source_2_root)
    source_1_root = remove_duplicate_nodes_in_branch(source_1_root)
    print("Deduplication complete.")

    p_json = save_nested_tree_to_json(source_2_root, source_2_out)
    v_json = save_nested_tree_to_json(source_1_root, source_1_out)
    print(f"Nested tree structures saved to {p_json} and {v_json}")


if __name__ == "__main__":
    # Example: search for a company in the source_1 tree
    source_1_company = "Christian Dior"
    print(f"\nSearching for '{source_1_company}' in source_1 tree...")
    search_and_print_company(
        "source_1_nested_tree.json",
        source_1_company,
        "christian_dior_source_1_tree.json",
        save_subtree=True,
    )
