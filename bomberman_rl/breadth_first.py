def bread_first_search(game_state, start, targets):
    def get_neighbors(node, visited_nodes):
        neighs = []
        if node[0]+1 <16: 
            neighs.append([node[0]+1,node[1]])
        if node[1]+1 <16:     
            neighs.append([node[0],node[1]+1])
        if node[0]-1 >0:
            neighs.append([node[0]-1,node[1]])
        if node[1]-1 >0:
            neighs.append([node[0],node[1]-1])
        valid_neighs = [neigh for neigh in neighs if game_state[neigh[0], neigh[1]] \
            and neigh not in visited_nodes]
        return valid_neighs
    all_nodes = [start]
    visited_nodes = [start]
    while all_nodes !=[]: 
        current = all_nodes.pop()
        visited_nodes.append(current)
        if current in targets: 
            return current
        neighs = get_neighbors(current, visited_nodes)
        all_nodes.extend(neighs)
    return False
