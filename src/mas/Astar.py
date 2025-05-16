import heapq
from vs.abstract_agent import AbstAgent

class Node:
    def __init__(self, position, parent=None, g=0, h=0):
        self.position = position  # (x, y)
        self.parent = parent
        self.g = g  # custo do início até aqui
        self.h = h  # heurística até o objetivo
        self.f = g + h  # custo total

    def __lt__(self, other):
        return self.f < other.f

class Astar:
    def __init__(self, map_obj=None, explorer=None, cost_line=1.0, cost_diag=1.5):
        """
        map_obj: instância de Map preenchida pelo Explorer
        cost_line: custo para andar em linha reta
        cost_diag: custo para andar na diagonal
        """
        self.map = map_obj
        self.explorer = explorer
        self.cost_line = cost_line
        self.cost_diag = cost_diag
        self.mapReturn = {}
        
    def set_map(self, map_obj):
        """
        Atualiza o mapa atual.
        map_obj: instância de Map preenchida pelo Explorer
        """
        self.map = map_obj
        
    def get_min_neighbor(self, pos):
        """
        Retorna a célula vizinha com menor custo.
        Se não houver vizinhos, retorna None.
        """
        neighbors = self.get_neighbors(pos)
        if not neighbors:
            return None
        return min(neighbors, key=lambda x: x[1])
        
    def set_difficulty(self, position, difficulty):
        """
        Heurística adaptada para mapas parcialmente conhecidos.
        Se houver obstáculos conhecidos no caminho direto, penaliza a heurística.
        """
        x, y = position

        # Primeiro calcula a dificuldade baseado no menor vizinho
        min_neighbor = self.get_min_neighbor((x, y))
        dif = min_neighbor[1] + difficulty
        result = dif
        if (x, y) in self.mapReturn and self.mapReturn[(x, y)][0] < dif:
            result = self.mapReturn[(x, y)][0]
            
        self.mapReturn[(x, y)] = [result, min_neighbor[0]]

    def heuristic(self, start, end):
        """
        Heuristic based on the Euclidean distance multiplied by the average difficulty of known cells.
        This approximates the real cost of returning to the base considering the already explored terrain.
        """
        x0, y0 = start
        x1, y1 = end

        # Euclidean distance (allows diagonals)
        dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5

        # Calculate the average difficulty of known cells (excluding obstacles)
        difficulties = []
        for pos in self.map.data:
            difficulty = self.map.get_difficulty(pos)
            if difficulty is not None and difficulty < 100:  # 100 = impassable obstacle
                difficulties.append(difficulty)
        if difficulties:
            avg_difficulty = sum(difficulties) / len(difficulties)
        else:
            avg_difficulty = 1  # default if no data

        return dist * avg_difficulty
        
        
    
    def get_neighbors(self, pos):
        # 8 direções: ortogonais e diagonais
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = pos[0] + dx, pos[1] + dy
            if self.map.in_map((nx, ny)):
                # Só considera célula se não for obstáculo conhecido e dificuldade aceitável
                difficulty = self.map.get_difficulty((nx, ny))
                if difficulty is not None:
                    neighbors.append(((nx, ny), difficulty*self.cost_diag if dx != 0 and dy != 0 else difficulty*self.cost_line))
        return neighbors

    def search(self, start, goal):
        """
        start: (x, y)
        goal: (x, y)
        Retorna lista de posições [(x1, y1), ...] ou None se não houver caminho.
        """
        
        # print(f"Explorador: {self.explorer.get_name()} OI")
                
        # Fila de prioridade (Heap mínimo) para os nós a serem explorados 
        open_set = []
        heapq.heappush(open_set, Node(start, None, 0, self.heuristic(start, goal)))
        
        # Conjunto de nós já explorados
        closed_set = set()
        
        # Dicionário para armazenar os custos g (custo do início até o nó)
        g_scores = {start: 0}
        
        # print(f"Explorador: {self.explorer.get_name()} COMEÇANDO")
        
        # print(f"Start: {start}, Goal: {goal}")

        while open_set:
            # if self.explorer.get_name == "EXPL_1":
            #     print(f"Explorador: {self.explorer.get_name()} - A* - Open set: {len(open_set)}, Closed set: {len(closed_set)}")
            current = heapq.heappop(open_set)
            if current.position == goal:
                return self.reconstruct_path(current)
            closed_set.add(current.position)
            for (neighbor_pos, step_cost) in self.get_neighbors(current.position):
                if neighbor_pos in closed_set:
                    continue
                # Cálculo do custo g (custo do início até nó atual)
                tentative_g = current.g + step_cost
                if neighbor_pos not in g_scores or tentative_g < g_scores[neighbor_pos]:
                    g_scores[neighbor_pos] = tentative_g
                    h = self.heuristic(neighbor_pos, goal)
                    neighbor_node = Node(neighbor_pos, current, tentative_g, h)
                    heapq.heappush(open_set, neighbor_node)
        return None

    def reconstruct_path(self, node):
        path = []
        while node:
            path.append((node.position, node.g))
            node = node.parent
        return path[::-1]
    
    