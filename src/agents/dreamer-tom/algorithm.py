import networkx as nx

# This is the complete content for the 'algorithm.py' file.

# --- Helper Functions (from original submission) ---

def _calculate_score(d1, d2):
    """Calculates the objective score for two dominating sets."""
    # The score is the size of the larger set plus the size of the intersection.
    return max(len(d1), len(d2)) + len(set(d1).intersection(set(d2)))

def _is_redundant(g, node_to_check, dom_set):
    """
    Checks if a node is redundant in a dominating set. A node is redundant
    if all nodes in its closed neighborhood are also dominated by other
    nodes in the set.
    """
    # The set of nodes that 'node_to_check' is responsible for dominating.
    nodes_to_check_coverage = set(g.neighbors(node_to_check)) | {node_to_check}
    other_dominators = dom_set - {node_to_check}

    for node in nodes_to_check_coverage:
        # Check if any node in the coverage set is left uncovered by other dominators.
        # A node 'n' is covered if its closed neighborhood intersects with the dominating set.
        if not (set(g.neighbors(node)) | {node}).intersection(other_dominators):
            # If we find even one node that becomes uncovered, then 'node_to_check' is NOT redundant.
            return False
    # If all nodes in the coverage set are still covered, 'node_to_check' is redundant.
    return True

def _get_private_neighbors(g, node, dom_set):
    """
    Returns the set of private neighbors of a node w.r.t. a dominating set.
    A node x is a private neighbor of 'node' if 'node' is the only dominator of x.
    """
    pn = set()
    # Check the node itself and all its direct neighbors.
    nodes_to_check = set(g.neighbors(node)) | {node}
    for n in nodes_to_check:
        # If the only dominator of 'n' is 'node', then 'n' is a private neighbor of 'node'.
        if (set(g.neighbors(n)) | {n}).intersection(dom_set) == {node}:
            pn.add(n)
    return pn


# --- Phase 1, Part A: Preprocessing (Handles trivial cases) ---

def _preprocess_graph(g):
    """
    Handles easy-to-solve parts of the graph to reduce problem complexity.
    Specifically, it identifies nodes with degree 1, as their single neighbor
    *must* be in any dominating set to cover them.
    """
    d_forced = set()
    nodes_to_process = set(g.nodes())
    
    # Find all nodes with degree 1 (leaves of the graph)
    degree_one_nodes = {n for n, deg in g.degree() if deg == 1 and n in nodes_to_process}

    if not degree_one_nodes:
        return set(), set(), set(g.nodes()), set(g.nodes())

    # The single neighbor of a degree-1 node must be chosen.
    for node in degree_one_nodes:
        # This was the source of the error. list(g.neighbors(node)) returns a list, e.g., [7].
        # We need the element itself, not the list containing it.
        neighbor = list(g.neighbors(node))[0]
        d_forced.add(neighbor)

    # Initial dominating sets start with these forced nodes
    d1 = set(d_forced)
    d2 = set(d_forced)

    # Update uncovered sets: remove nodes already covered by the forced nodes
    u1 = set(g.nodes())
    u2 = set(g.nodes())
    
    newly_covered = set()
    for node in d_forced:
        newly_covered.update(set(g.neighbors(node)) | {node})
        
    u1.difference_update(newly_covered)
    u2.difference_update(newly_covered)
    
    return d1, d2, u1, u2

# --- Phase 1, Part B: SLL-Inspired Greedy Construction ---

def _sll_greedy_construction(g, d1_init, d2_init, u1_init, u2_init):
    """
    Constructs two dominating sets D1 and D2 simultaneously using an advanced
    greedy heuristic inspired by SLL (Set-Cover-based Local Levy). This
    heuristic prioritizes covering nodes that have fewer options to be covered.
    """
    d1, d2 = set(d1_init), set(d2_init)
    u1, u2 = set(u1_init), set(u2_init)
    
    # Pre-calculate weights for all nodes. Weight is 1/degree.
    # This gives priority to covering low-degree nodes.
    weights = {n: 1.0 / (g.degree(n) + 1) for n in g.nodes()}

    while u1 or u2:
        best_move = (-1, -1, None, None)  # (score, -node_id, node, target_set)

        # Build a set of candidate nodes to evaluate.
        # Candidates are nodes that can cover at least one uncovered node.
        candidates = set()
        for node_set in [u1, u2]:
            for node in node_set:
                candidates.update(g.neighbors(node))
                candidates.add(node)
        
        for v in candidates:
            # Evaluate adding node 'v' to D1
            if u1:
                # Benefit is the sum of weights of newly covered nodes
                benefit1 = sum(weights[n] for n in (set(g.neighbors(v)) | {v}) if n in u1)
                if benefit1 > 0:
                    # Cost is higher if 'v' is already in the other set
                    cost1 = 1.0 if v not in d2 else 2.0 
                    score1 = benefit1 / cost1
                    move1 = (score1, -v, v, 'd1')
                    if move1 > best_move: best_move = move1

            # Evaluate adding node 'v' to D2
            if u2:
                benefit2 = sum(weights[n] for n in (set(g.neighbors(v)) | {v}) if n in u2)
                if benefit2 > 0:
                    cost2 = 1.0 if v not in d1 else 2.0
                    score2 = benefit2 / cost2
                    move2 = (score2, -v, v, 'd2')
                    if move2 > best_move: best_move = move2
        
        if best_move[2] is None: # No beneficial move found
            if u1: d1.update(u1) # Add all remaining uncovered nodes if any
            if u2: d2.update(u2)
            break

        _, _, node_to_add, target_set_str = best_move
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
    # Iterate in a fixed order (by decreasing degree, then node ID) for determinism
    for node in sorted(list(dom_set), key=lambda n: (-g.degree(n), n)):
        if node in d_minimal and _is_redundant(g, node, d_minimal):
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
    
    max_iterations = 2000 # A simple iteration cap to respect the time limit
    
    for _ in range(max_iterations):
        current_score = _calculate_score(list(d1_r), list(d2_r))
        improved = False

        # Move 1: Reduce Intersection (high-impact move)
        # Try to remove a node from one set if it's redundant.
        for node in sorted(list(d1_r.intersection(d2_r))):
            if _is_redundant(g, node, d1_r):
                d1_r.remove(node); improved = True; break
            if _is_redundant(g, node, d2_r):
                d2_r.remove(node); improved = True; break
        if improved: continue

        # Move 2: 1-Swaps (exchange a node in D with a node not in D)
        # For D1
        for u in sorted(list(d1_r)):
            pn_u = _get_private_neighbors(g, u, d1_r)
            if not pn_u: continue # Cannot swap out a node with no private neighbors easily
            
            # Find a node 'v' outside d1_r that can cover u's private neighbors
            for v in sorted(list(all_nodes - d1_r)):
                if pn_u.issubset(set(g.neighbors(v)) | {v}):
                    d1_temp = (d1_r - {u}) | {v}
                    if _calculate_score(list(d1_temp), list(d2_r)) < current_score:
                        d1_r = d1_temp; improved = True; break
            if improved: break
        if improved: continue

        # For D2
        for u in sorted(list(d2_r)):
            pn_u = _get_private_neighbors(g, u, d2_r)
            if not pn_u: continue
            
            for v in sorted(list(all_nodes - d2_r)):
                if pn_u.issubset(set(g.neighbors(v)) | {v}):
                    d2_temp = (d2_r - {u}) | {v}
                    if _calculate_score(list(d1_r), list(d2_temp)) < current_score:
                        d2_r = d2_temp; improved = True; break
            if improved: break
        if improved: continue

        if not improved:
            break
            
    return d1_r, d2_r

# --- Main Function for the Competition ---

def dominant(g):
    """
    Finds two dominating sets D1 and D2 for a graph g to minimize
    the objective function: max(|D1|, |D2|) + |D1 intersect D2|.

    The algorithm is deterministic and follows a multi-phase heuristic approach:
    1.  Preprocessing: Handles trivial cases like degree-1 nodes to simplify
        the graph.
    2.  Construction: An advanced greedy algorithm (SLL-inspired) builds an
        initial high-quality solution pair.
    3.  Refinement: A deterministic local search iteratively improves the solution
        by making small, score-improving changes.
    4.  Purification: A final cleanup ensures the output sets are minimal.

    :param g: the graph as a networkx structure.
    """
    if g.number_of_nodes() == 0:
        return [], []

    # Phase 1: Preprocessing and Greedy Construction
    d1_init, d2_init, u1_init, u2_init = _preprocess_graph(g)
    d1, d2 = _sll_greedy_construction(g, d1_init, d2_init, u1_init, u2_init)

    # Phase 2 & 3: Purification and Refinement
    d1 = _purify(g, d1)
    d2 = _purify(g, d2)
    
    # Run local search only if the graph is not too large to avoid timeouts
    if g.number_of_nodes() < 2000:
       d1, d2 = _local_search_refinement(g, d1, d2)

    # Final purification to ensure minimality after local search
    d1 = _purify(g, d1)
    d2 = _purify(g, d2)

    # Return as sorted lists for deterministic output
    return sorted(list(d1)), sorted(list(d2))
