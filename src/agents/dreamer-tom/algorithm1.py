import networkx as nx

def dominant(g):
    """
    Finds two dominating sets, D1 and D2, with a minimal intersection using a
    two-phase approach: Greedy Construction + Local Improvement.

    Phase 1: Greedy Construction
    - D1 is generated with a greedy algorithm that prioritizes covering the most nodes.
    - D2 is generated similarly but with a heavy penalty for choosing nodes from D1.
    
    Phase 2: Local Improvement
    - Both D1 and D2 are "pruned" by a refinement step.
    - This step iterates through each node in the set and removes it if it's redundant
      (i.e., if the set remains a valid dominating set without it).
    - This process repeats until no more nodes can be removed, resulting in a smaller,
      minimal dominating set.

    :param g: The graph as a NetworkX object.
    :return: A tuple of two lists of nodes (D1, D2).
    """

    def _greedy_search(graph, penalized_nodes=None):
        """
        Phase 1: Constructs an initial dominating set using a greedy heuristic.
        """
        if penalized_nodes is None:
            penalized_nodes = set()
            
        dominating_set = set()
        uncovered_nodes = set(graph.nodes())
        penalty = float(len(graph.nodes()) + 1)
        
        # Pre-calculate neighbors for efficiency
        neighbors_map = {node: set(graph.neighbors(node)) | {node} for node in graph.nodes()}
        
        while uncovered_nodes:
            best_node = -1
            max_score = -1.0
            
            # Find the node that covers the most new nodes
            candidate_nodes = [n for n in graph.nodes() if n not in dominating_set]
            for node in candidate_nodes:
                score = float(len(neighbors_map[node].intersection(uncovered_nodes)))
                
                if node in penalized_nodes:
                    score -= penalty
                
                if score > max_score:
                    max_score = score
                    best_node = node
                elif score == max_score:
                    if best_node == -1 or node < best_node:
                        best_node = node
            
            if best_node == -1:
                if not uncovered_nodes: break
                best_node = uncovered_nodes.pop()

            dominating_set.add(best_node)
            uncovered_nodes -= neighbors_map[best_node]
            
        return dominating_set

    def _local_improvement(graph, initial_ds):
        """
        Phase 2: Refines a dominating set by removing redundant nodes.
        """
        ds = set(initial_ds)
        
        # Pre-calculate neighbors map for the whole graph for faster lookups
        neighbors_map = {node: set(graph.neighbors(node)) | {node} for node in graph.nodes()}

        nodes_to_check = sorted(list(ds), key=lambda n: graph.degree(n))

        for node_to_remove in nodes_to_check:
            if node_to_remove not in ds:
                continue

            # Check if this node is essential
            potential_ds = ds - {node_to_remove}
            
            # Find all nodes that node_to_remove was covering
            nodes_that_were_covered = neighbors_map[node_to_remove]
            
            is_redundant = True
            for node in nodes_that_were_covered:
                # Is this node still covered by the rest of the set?
                is_still_covered = any(node in neighbors_map[dominator] for dominator in potential_ds)
                
                if not is_still_covered:
                    is_redundant = False
                    break
            
            # If it was redundant, remove it for real
            if is_redundant:
                ds.remove(node_to_remove)
                
        return list(ds)

    # --- Main execution flow ---

    # 1. Construct initial sets using the penalized greedy algorithm
    d1_initial = _greedy_search(g)
    d2_initial = _greedy_search(g, penalized_nodes=d1_initial)

    # 2. Refine both sets by removing redundant nodes
    d1_final = _local_improvement(g, d1_initial)
    d2_final = _local_improvement(g, d2_initial)
    
    return d1_final, d2_final
    