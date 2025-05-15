import heapq
from vs.abstract_agent import AbstAgent
from vs.constants import VS

class ComeBack:
    def __init__(self, cost_line=1.0, cost_diag=1.5):
        """
        map_obj: instância de Map preenchida pelo Explorer
        cost_line: custo para andar em linha reta
        cost_diag: custo para andar na diagonal
        """
        self.cost_line = cost_line
        self.cost_diag = cost_diag
        self.mapReturn = {}
        self.mapReturn[(0, 0)] = 0

    def heuristic(self, start, difficulty):
        """
        Heurística adaptada para mapas parcialmente conhecidos.
        Se houver obstáculos conhecidos no caminho direto, penaliza a heurística.
        """
        x, y = start

        # Primeiro calcula a dificuldade baseado no menor vizinho
        dif = self.get_min_neighbor((x, y)) + difficulty
        result = dif
        if (x, y) in self.mapReturn and self.mapReturn[(x, y)] < dif:
            result = self.mapReturn[(x, y)]
            
        self.mapReturn[(x, y)] = result
        return result
    
    def get_neighbors(self, pos):
        # 8 direções: ortogonais e diagonais
        directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
        neighbors = []
        for dx, dy in directions:
            nx, ny = pos[0] + dx, pos[1] + dy
            if (nx, ny) in self.mapReturn:
                # Só considera célula se não for obstáculo conhecido e dificuldade aceitável
                difficulty = self.mapReturn[(nx, ny)]
                if difficulty is not None:
                    neighbors.append(((nx, ny), difficulty*self.cost_diag if dx != 0 and dy != 0 else difficulty*self.cost_line))
        return neighbors
    
    def get_min_neighbor(self, pos):
        """
        Retorna a célula vizinha com menor custo.
        Se não houver vizinhos, retorna None.
        """
        neighbors = self.get_neighbors(pos)
        if not neighbors:
            return None
        return min(neighbors, key=lambda x: x[1])[1]