from collections import defaultdict
from typing import Dict, List, Set, Tuple
from collections import Counter

Chains = List[List[int]]

def ratio_of_len2_chains(chains: Chains) -> float:
    total = len(chains)
    if total == 0:
        return 2.0
    count_len2 = sum(len(c) == 2 for c in chains)
    return count_len2 / total

def compute_chains_endpoint_degrees(chains: Chains) -> dict[int, int]:
    """
    compute the degree of each endpoint in the chains
    """
    deg_counter = Counter()
    for chain in chains:
        if not chain:
            continue
        start, end = chain[0], chain[-1]
        deg_counter[start] += 1
        deg_counter[end] += 1
    return dict(deg_counter)

def sort_chains(chains: Chains) -> Chains:
    def sort_chain(chain: List[int]) -> List[int]:
        # sort the chain by the first and last point
        if chain[0] > chain[-1]:
            chain = chain[::-1]
        elif chain[0] == chain[-1]:
            chain = chain[:-1]
            min_value = min(chain)
            min_index = chain.index(min_value)
            chain = chain[min_index:] + chain[:min_index]
            chain.append(chain[0])
        return chain

    # sort each chain
    chains = [sort_chain(chain) for chain in chains]
    
    # sort the chains by the first point
    chains.sort(key=lambda x: x[0])
    
    return chains

def sort_chains_by_length(chains: Chains) -> Chains:
    # def sort_chain(chain: List[int]) -> List[int]:
    #     # sort the chain by the first and last point
    #     if chain[0] > chain[-1]:
    #         chain = chain[::-1]
    #     elif chain[0] == chain[-1]:
    #         chain = chain[:-1]
    #         min_value = min(chain)
    #         min_index = chain.index(min_value)
    #         chain = chain[min_index:] + chain[:min_index]
    #         chain.append(chain[0])
    #     return chain

    # sort each chain
    # chains = [sort_chain(chain) for chain in chains]
    
    # sort the chains by the length and the first point
    chains.sort(key=lambda x: (-len(x), x[0]))
    
    return chains


def sort_and_deduplicate_chains(chains: Chains) -> Chains:
    def sort_chain(chain: List[int]) -> List[int]:
        # sort the chain by the first and last point
        if chain[0] > chain[-1]:
            chain = chain[::-1]
            chain = list(dict.fromkeys(chain)) # 去除重复点
        elif chain[0] == chain[-1]:
            chain = chain[:-1]
            chain = list(dict.fromkeys(chain)) # 去除重复点
            
            if degrees[chain[0]] > 2:
                # 说明该 loop 和其他 chain 是相连的，因此其起点和终点是固定的
                chain.append(chain[0])
                return chain
            
            min_value = min(chain)
            min_index = chain.index(min_value)
            chain = chain[min_index:] + chain[:min_index]
            chain.append(chain[0])
        return chain

    degrees = compute_chains_endpoint_degrees(chains)

    seen = set()
    unique_chains = []

    for chain in chains:
        chain_tuple = tuple(sort_chain(chain))
        if chain_tuple not in seen:
            seen.add(chain_tuple)
            unique_chains.append(list(chain_tuple))

    # unique_chains.sort(key=lambda x: x[0])
    # unique_chains.sort(key=lambda x: (len(x), x[0]))
    # unique_chains = sort_chains_by_length(unique_chains)
    
    unique_chains.sort(key=lambda x: (-len(x), x[0]))
    

    return unique_chains


def split_graph_into_chains(adjacency_list) -> Chains:
    """
    Split an undirected graph into chains whose internal nodes all have degree 2.
    Each chain begins/ends at a node with degree ≠ 2 **or** forms a pure cycle.

    Parameters
    ----------
    adjacency_list : list[(u, v)]
        Undirected edges. Node indices are kept untouched.

    Returns
    -------
    chains : list[list[int]]
        A list of chains (vertex sequences).  Order along each chain is the
        traversal order, useful for downstream re-indexing.
    """
    # ---------- build adjacency ----------
    adj: Dict[int, List[int]] = defaultdict(list)
    for u, v in adjacency_list:
        adj[u].append(v)
        adj[v].append(u)

    # ---------- helpers ----------
    def is_degree_two(n: int) -> bool:
        return len(adj[n]) == 2

    visited_edge: Set[Tuple[int, int]] = set()   # directed edge mark
    chains: Chains = []

    # ---------- first pass: start from non-deg-2 nodes ----------
    for start in adj:
        if is_degree_two(start):
            continue

        for nxt in adj[start]:
            if (start, nxt) in visited_edge:
                continue

            chain = [start]
            prev, cur = start, nxt

            while True:
                # mark directed edge as used
                visited_edge.add((prev, cur))
                visited_edge.add((cur, prev))
                chain.append(cur)

                if not is_degree_two(cur):                # reached another fork/end
                    break

                # choose the other neighbor (degree=2 ⇒ exactly 2 neighbors)
                n1, n2 = adj[cur]
                nxt = n1 if n1 != prev else n2

                if (cur, nxt) in visited_edge:            # loop already visited
                    break

                prev, cur = cur, nxt

            if len(chain) > 1:                            # ignore isolated forks
                chains.append(chain)

    # ---------- second pass: leftover pure cycles ----------
    unvisited_deg2 = [n for n in adj if is_degree_two(n) 
                      and all((n, nb) not in visited_edge for nb in adj[n])]

    # sort the unvisited_deg2
    unvisited_deg2.sort()

    for start in unvisited_deg2:
        if any((start, nb) in visited_edge for nb in adj[start]):
            continue  # already handled in another cycle pass

        chain = [start]
        prev, cur = start, adj[start][0]

        while True:
            visited_edge.add((prev, cur))
            visited_edge.add((cur, prev))
            chain.append(cur)

            n1, n2 = adj[cur]              # two neighbors
            nxt = n1 if n1 != prev else n2

            if nxt == start:               # cycle closed
                break
            prev, cur = cur, nxt

        # Ensure the cycle is closed by adding the start node to the end
        if chain[0] != chain[-1]:
            chain.append(chain[0])  # Close the loop by adding the start point at the end

        chains.append(chain)


    # chains = sort_chains(chains)
    # chains = sort_chains_by_length(chains)

    return chains


def split_and_filter_chains_1D(chains_1D: List[int]) -> Chains:
    chains = []
    temp_chain = []
    
    element_counts = Counter(chains_1D)
    
    for vtx_idx in chains_1D:
        if vtx_idx == -1:
            if len(temp_chain) == 2:
                if element_counts[temp_chain[0]] == 1 or element_counts[temp_chain[1]] == 1:
                    temp_chain = []
                    continue            
            if temp_chain:
                chains.append(temp_chain)
            temp_chain = []
        else:
            temp_chain.append(vtx_idx)
    
    if len(temp_chain) == 2:
        # If a chain (length=2) has a degree-1 endpoint, it'd be a spike and can be removed.
        if element_counts[temp_chain[0]] == 1 or element_counts[temp_chain[1]] == 1:
            temp_chain = []
        else:
            chains.append(temp_chain)
    elif temp_chain:
        chains.append(temp_chain)
    
    # chains = sort_chains(chains)
    # chains = sort_chains_by_length(chains)
    
    return chains


def filter_chains(chains: Chains) -> Chains:
    """"
    Filter burr and float chains
    burr chains: chains with length 2 and one of the nodes has degree 1
    float chains: chains with length 2 and both nodes have degree 1
    """
    
    all_nodes = [node for chain in chains for node in chain]
    element_counts = Counter(all_nodes)
    
    filtered_chains = []
    
    for chain in chains:
        if len(chain) == 2:
            if element_counts[chain[0]] == 1 or element_counts[chain[1]] == 1:
                continue
        
        filtered_chains.append(chain)
    
    return filtered_chains


def extract_seams_by_faces(faces):
    edge_uv_pairs = defaultdict(set)
    
    for face in faces:
        # face: [v0, v1, v2, vt0, vt1, vt2]
        for i in range(3):
            v_i, vt_i = face[i], face[i+3]
            v_j, vt_j = face[(i + 1) % 3], face[(i + 1) % 3 + 3]
            edge_key = frozenset({v_i, v_j})                 # 无向几何边
            uv_pair   = frozenset({vt_i, vt_j})              # 无向 UV 边
            edge_uv_pairs[edge_key].add(uv_pair)

    seam_edges = [tuple(edge) for edge, uvset in edge_uv_pairs.items()
                if len(uvset) > 1]                        # 多种 UV 组合 ⇒ seam

    return seam_edges

def flatten_and_add_marker(chains: Chains):
    chains_1D = []
    flags = []
    
    for chain in chains:
        start, end = chain[0], chain[-1]
        # 这里我们进行了一次倒序，是为了容易进行算法实现。现在的 Sequence 是 [start, end, p1, p2, ..., pN, -1], p1-N 是从 end 指向 start 的。
        # 其中 -1 是一个特殊标记，表示该 chain 的结束。
        new_seq = [start] + [end] + chain[1:-1][::-1] + [-1]
        tmp_mask = [0] + [1] + [2] * (len(chain) - 2) + [-1]
        
        chains_1D.extend(new_seq)
        flags.extend(tmp_mask)
    
    chains_1D[-1] = -2
    flags[-1] = -2
    
    # return [chains_1D, flags]
    assert len(chains_1D) == len(flags), "chains_1D and flags must have the same length"
    
    return {
        'chains_1D': chains_1D,
        'flags': flags,
    }
