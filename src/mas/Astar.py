import heapq
import math

# Classe que representa um nó no grafo do A*
class Node:
    def __init__(self, position, parent=None, g=0, h=0):
        self.position = position  # Coordenada (x, y)
        self.parent = parent      # Pai do nó atual (usado para reconstruir o caminho)
        self.g = g                # Custo do início até este nó
        self.h = h                # Heurística do nó até o objetivo
        self.f = g + h            # Custo total estimado (f = g + h)

    def __lt__(self, other):
        return self.f < other.f  # Define ordenação com base no custo total

# Classe que implementa o algoritmo A*
class Astar:
    def __init__(self, map_obj, cost_line=1.0, cost_diag=1.5):
        self.map = map_obj              # Mapa do ambiente (deve implementar métodos in_map e get_difficulty)
        self.cost_line = cost_line      # Custo para mover em linha reta
        self.cost_diag = cost_diag      # Custo para mover na diagonal

    # Heurística de distância Manhattan (boa para grids com movimentação 4 ou 8 direções)
    def heuristic(self, start, end):
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        return self.cost_line * (dx + dy)

    # Retorna os vizinhos válidos de uma posição no mapa
    def get_neighbors(self, pos):
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = pos[0] + dx, pos[1] + dy
            if self.map.in_map((nx, ny)):
                difficulty = self.map.get_difficulty((nx, ny))
                # Verifica se a célula é conhecida e não é um obstáculo intransponível
                if difficulty is not None and difficulty < 100:
                    cost = difficulty * (self.cost_diag if dx != 0 and dy != 0 else self.cost_line)
                    neighbors.append(((nx, ny), cost))
        return neighbors

    # Algoritmo principal do A*, com suporte a verificação de bateria restante
    def search(self, start, goal, battery_limit=None):
        open_set = []  # Fila de prioridade (heap)
        heapq.heappush(open_set, Node(start, None, 0, self.heuristic(start, goal)))
        closed_set = set()  # Conjunto de posições já exploradas
        g_scores = {start: 0}  # Custo acumulado até cada nó visitado

        while open_set:
            current = heapq.heappop(open_set)

            # Ignora caminhos que excedem o limite de bateria
            if battery_limit is not None and current.g > battery_limit:
                continue

            # Caminho encontrado até o objetivo
            if current.position == goal:
                return self.reconstruct_path(current)

            closed_set.add(current.position)

            for (neighbor_pos, step_cost) in self.get_neighbors(current.position):
                if neighbor_pos in closed_set:
                    continue

                tentative_g = current.g + step_cost

                if battery_limit is not None and tentative_g > battery_limit:
                    continue  # Evita expandir caminhos inviáveis

                if neighbor_pos not in g_scores or tentative_g < g_scores[neighbor_pos]:
                    g_scores[neighbor_pos] = tentative_g
                    h = self.heuristic(neighbor_pos, goal)
                    neighbor_node = Node(neighbor_pos, current, tentative_g, h)
                    heapq.heappush(open_set, neighbor_node)

        return None  # Nenhum caminho encontrado

    # Reconstrói o caminho a partir do nó final até o início
    def reconstruct_path(self, node):
        path = []
        while node:
            path.append(node.position)
            node = node.parent
        return path[::-1]  # Retorna o caminho do início até o fim
