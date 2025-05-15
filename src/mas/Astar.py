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
        
    def set_map(self, map_obj):
        """
        Atualiza o mapa atual.
        map_obj: instância de Map preenchida pelo Explorer
        """
        self.map = map_obj

    def heuristic(self, start, end):
        """
        Heurística adaptada para mapas parcialmente conhecidos.
        Se houver obstáculos conhecidos no caminho direto, penaliza a heurística.
        """
        x0, y0 = start
        x1, y1 = end
        
        # Valor da heurística inicial
        difficulty = 0
        
        # Calcula os catetos do triângulo retângulo formado pelos pontos de início e fim
        deltaX = x1 - x0
        deltaY = y1 - y0
        
        # Calcula a direção do movimento
        if deltaX != 0:
            dx = deltaX/abs(deltaX)
        else:
            dx = 0
        if deltaY != 0:
            dy = deltaY/abs(deltaY)
        else:
            dy = 0
            
        # Inicia a posição corrente
        x = x0
        y = y0
        
        # Enquanto não chegar ao destino, calcula a dificuldade até o destino
        while (x, y) != (x1, y1):
            pos_x = x + dx
            pos_y = y + dy
            data_pos = self.map.get((pos_x, pos_y))
            if data_pos is not None:
                
                # Soma quanto vai custar para andar até a célula
                if dx != 0 and dy != 0:
                    difficulty += data_pos[0]*self.cost_diag
                else:
                    difficulty += data_pos[0]*self.cost_line
                
                # Atualiza a posição corrente e a direção do movimento
                x, y = pos_x, pos_y
                deltaX = x1 - pos_x
                deltaY = y1 - pos_y
                if deltaX != 0:
                    dx = deltaX/abs(deltaX)
                else:
                    dx = 0
                if deltaY != 0:
                    dy = deltaY/abs(deltaY)
                else:
                    dy = 0
            else:
                dir = 0
                match dx, dy:
                    case 1, -1:
                        dir = 1
                    case 1, 0:
                        dir = 2
                    case 1, 1:
                        dir = 3
                    case 0, 1:
                        dir = 4
                    case -1, 1:
                        dir = 5
                    case -1, 0:
                        dir = 6
                    case -1, -1:
                        dir = 7
                dx, dy = AbstAgent.AC_INCR[dir]
        
        return difficulty
    

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
        
        if self.explorer.get_name == "EXPL_1":
                print(f"Explorador: {self.explorer.get_name()} OI")
                
        # Fila de prioridade (Heap mínimo) para os nós a serem explorados 
        open_set = []
        heapq.heappush(open_set, Node(start, None, 0, self.heuristic(start, goal)))
        
        # Conjunto de nós já explorados
        closed_set = set()
        
        # Dicionário para armazenar os custos g (custo do início até o nó)
        g_scores = {start: 0}
        
        if self.explorer.get_name == "EXPL_1":
                print(f"Explorador: {self.explorer.get_name()} COMEÇANDO")
        
        # print(f"Start: {start}, Goal: {goal}")

        while open_set:
            if self.explorer.get_name == "EXPL_1":
                print(f"Explorador: {self.explorer.get_name()} - A* - Open set: {len(open_set)}, Closed set: {len(closed_set)}")
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
            path.append(node.position)
            node = node.parent
        return path[::-1]
    
    