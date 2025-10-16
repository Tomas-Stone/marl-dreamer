import networkx as nx

# This is the complete content for the 'algorithm.py' file.

# --- Helper Functions ---

def _calculate_score(d1, d2):
    """Calculates the objective score for two dominating sets."""
    return max(len(d1), len(d2)) + len(d1.intersection(d2))

def _get_uncovered_neighbors_count(g, node, uncovered_set):
    """Efficiently counts how many nodes in uncovered_set are covered by 'node'."""
    count = 0
    if node in uncovered_set:
        count += 1
    # Iterating is faster than creating a new set for N[node]
    for neighbor in g.neighbors(node):
        if neighbor in uncovered_set:
            count += 1
    return count

def _is_redundant(g, node_to_check, dom_set):
    """
    Checks if a node is redundant in a dominating set. A node is redundant
    if all nodes in its closed neighborhood are also dominated by other
    nodes in the set. This is much faster than a full graph check.
    """
    nodes_to_check_coverage = set(g.neighbors(node_to_check)) | {node_to_check}
    other_dominators = dom_set - {node_to_check}
    for node in nodes_to_check_coverage:
        # If any node that was covered by node_to_check is NOT covered by the rest
        if not (set(g.neighbors(node)) | {node}) & other_dominators:
            return False
    return True

def _get_private_neighbors(g, node, dom_set):
    """
    Returns the set of private neighbors of a node w.r.t. a dominating set.
    A node x is a private neighbor of 'node' if 'node' is the only dominator of x.
    """
    pn = set()
    nodes_to_check = set(g.neighbors(node)) | {node}
    for n in nodes_to_check:
        if (set(g.neighbors(n)) | {n}) & dom_set == {node}:
            pn.add(n)
    return pn


# --- Phase 1: Integrated Greedy Construction ---

def _integrated_greedy_construction(g):
    """
    Constructs two dominating sets D1 and D2 simultaneously using a greedy
    heuristic. The heuristic's scoring function is tailored to the problem's
    objective, prioritizing covering new nodes while penalizing intersection.
    """
    nodes = list(g.nodes)
    d1, d2 = set(), set()
    u1, u2 = set(nodes), set(nodes)
    
    degrees = {n: g.degree(n) for n in nodes}

    while u1 or u2:
        best_move = (-1, -1, -1, None, None)  # (score, degree, -node_id, node, target_set)

        candidates = set()
        if u1:
            candidates.update(u1)
            for node in u1: candidates.update(g.neighbors(node))
        if u2:
            candidates.update(u2)
            for node in u2: candidates.update(g.neighbors(node))
        
        for v in candidates:
            if u1:
                benefit1 = _get_uncovered_neighbors_count(g, v, u1)
                if benefit1 > 0:
                    cost1 = 1.0 if v not in d2 else 2.0
                    score1 = benefit1 / cost1
                    move1 = (score1, degrees[v], -v, v, 'd1')
                    if move1 > best_move: best_move = move1

            if u2:
                benefit2 = _get_uncovered_neighbors_count(g, v, u2)
                if benefit2 > 0:
                    cost2 = 1.0 if v not in d1 else 2.0
                    score2 = benefit2 / cost2
                    move2 = (score2, degrees[v], -v, v, 'd2')
                    if move2 > best_move: best_move = move2
        
        if best_move[1] is None:
            if u1: d1.update(u1); u1.clear()
            if u2: d2.update(u2); u2.clear()
            break

        _, _, _, node_to_add, target_set_str = best_move
        newly_covered = (set(g.neighbors(node_to_add)) | {node_to_add})
        
        if target_set_str == 'd1':
            d1.add(node_to_add)
            u1.difference_update(newly_covered)
        else:
            d2.add(node_to_add)
            u2.difference_update(newly_covered)

    return d1, d2

# --- Phase 2: Purification ---

def _purify(g, dom_set):
    """
    Removes redundant nodes from a dominating set to make it minimal.
    Iterates through nodes and removes any that are not essential for domination.
    """
    d_minimal = set(dom_set)
    # Iterate in a fixed order (e.g., by decreasing degree) for determinism
    for node in sorted(list(dom_set), key=lambda n: (-g.degree(n), n)):
        if _is_redundant(g, node, d_minimal):
            d_minimal.remove(node)
    return d_minimal

# --- Phase 3: Local Search Refinement ---

def _local_search_refinement(g, d1, d2):
    """
    Improves a pair of dominating sets using a deterministic local search.
    It systematically evaluates a neighborhood of solutions (via swaps and
    removals) and applies the first move that improves the objective score.
    """
    d1_r, d2_r = set(d1), set(d2)
    all_nodes = set(g.nodes())
    
    # A simple iteration cap to respect the time limit
    max_iterations = 100 
    
    for _ in range(max_iterations):
        current_score = _calculate_score(d1_r, d2_r)
        improved = False

        # Move 1: Reduce Intersection (high-impact move)
        for node in sorted(list(d1_r & d2_r)):
            if _is_redundant(g, node, d1_r):
                d1_r.remove(node); improved = True; break
            if _is_redundant(g, node, d2_r):
                d2_r.remove(node); improved = True; break
        if improved: continue

        # Move 2: 1-Swaps (u in D, v not in D)
        # For D1
        for u in sorted(list(d1_r)):
            pn_u = _get_private_neighbors(g, u, d1_r)
            for v in sorted(list(all_nodes - d1_r)):
                if pn_u.issubset(set(g.neighbors(v)) | {v}):
                    d1_temp = (d1_r - {u}) | {v}
                    if _calculate_score(d1_temp, d2_r) < current_score:
                        d1_r = d1_temp; improved = True; break
            if improved: break
        if improved: continue

        # For D2
        for u in sorted(list(d2_r)):
            pn_u = _get_private_neighbors(g, u, d2_r)
            for v in sorted(list(all_nodes - d2_r)):
                if pn_u.issubset(set(g.neighbors(v)) | {v}):
                    d2_temp = (d2_r - {u}) | {v}
                    if _calculate_score(d1_r, d2_temp) < current_score:
                        d2_r = d2_temp; improved = True; break
            if improved: break
        if improved: continue

        if not improved: break
            
    return d1_r, d2_r

# --- Main Function for the Competition ---

def dominant(g):
    """
    Finds two dominating sets D1 and D2 for a graph g to minimize
    the objective function: max(|D1|, |D2|) + |D1 intersect D2|.

    The algorithm is deterministic and follows a multi-phase heuristic approach:
    1.  Construction: An integrated greedy algorithm builds an initial solution pair.
    2.  Refinement: A deterministic local search iteratively improves the solution.
    3.  Purification: A final cleanup ensures the output sets are minimal.

    :param g: the graph as a networkx structure.
    """
    if g.number_of_nodes() == 0:
        return [], []

    # Phase 1: Construction
    d1, d2 = _integrated_greedy_construction(g)

    # Phase 2: Refinement (with pre- and post-purification)
    d1 = _purify(g, d1)
    d2 = _purify(g, d2)
    
    d1, d2 = _local_search_refinement(g, d1, d2)

    # Phase 3: Final Purification
    d1 = _purify(g, d1)
    d2 = _purify(g, d2)

    return sorted(list(d1)), sorted(list(d2))