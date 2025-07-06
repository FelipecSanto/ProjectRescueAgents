# EXPLORER AGENT
# @Author: Tacla, UTFPR
#
### It walks randomly in the environment looking for victims. When half of the
### exploration has gone, the explorer goes back to the base.

import sys
import os
import random
import math
from abc import ABC, abstractmethod
from comeBack import ComeBack
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map
from Astar import Astar

class Explorer(AbstAgent):
    """ class attribute """
    MAX_DIFFICULTY = 1             # the maximum degree of difficulty to enter into a cell
    
    def __init__(self, env, config_file, resc, dir):
        """ Construtor do agente random on-line
        @param env: a reference to the environment 
        @param config_file: the absolute path to the explorer's config file
        @param resc: a reference to the rescuer agent to invoke when exploration finishes
        """

        super().__init__(env, config_file)
        self.walk_stack = []       # a stack to store the movements
        self.walk_time = 0         # time consumed to walk when exploring (to decide when to come back)
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals
        self.quadrante = dir - 1
        self.direction = (dir - 1) * 2
        self.lista = []
        self.goto = []
        self.min = 0, 0
        self.return_time = 0
        self.finishedQ = False
        self.finishedAll = False
        self.obstacles = {}
        self.dir_view = 0
        
        self.aStar = Astar(self.map, self)
        self.return_path = []
        self.it_return_path = 1
        self.last_path = True

        self.max_Difficulty = 1
        
        self.update_cont = 0

        # put the current position - the base - in the map
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())
        
    def get_name(self):
        """ Returns the name of the agent """
        return self.NAME

    def get_next_position(self):
        
        if self.goto:
            go = self.goto[0]
            self.goto.pop(0)
            return go

        # Direção inicial, baseado no quadrante
        self.dir_view = (self.direction + 4) % 8

        # Contador para evitar loops infinitos
        cont = 0
        cont_back = 0
        virtual_pos = self.x, self.y

        while True:
            cont += 1
            # Se o contador for maior que 8, significa que não há mais direções disponíveis
            # O agente tentará voltar pela pilha de movimentos
            # Ele ainda verificará se há alguma direção válida enquanto estiver voltando, e se houver, ele irá nessa direção
            if cont > 8:
                cont = 1
                cont_back += 1
                result = self.walk_stack[-cont_back]
                virtual_pos = virtual_pos[0] - result[0], virtual_pos[1] - result[1]
                if virtual_pos == (0,0):
                    if not self.finishedQ:
                        return self.aumenta_quadrante()
                    else:
                        self.goto_astar((0, 0))
                        self.finishedAll = True
                        return self.get_next_position()
            
            result = self.verifica_direcao(virtual_pos)
            if result:
                return result
                       
    def verifica_direcao(self, pos):
        self.obstacles[self.x,self.y] = self.check_walls_and_lim()

        # Inicializa dx e dy (direções que o agente vai andar)
        dx, dy = 0, 0

        # Verifica as direções e condições para cada quadrante
        if self.quadrante == 0:
            # Comentários apenas para o primeiro quadrante, já que os outros são semelhantes
            # Se a direção for a inicial, gira para a horário, senão gira para anti-horário
            # Isso porque depende se ele está indo ou voltando, e isso ajuda a sempre preencher onde está faltando
            if self.direction == 0:
                self.dir_view = (self.dir_view + 1) % 8
            else:
                self.dir_view = (self.dir_view - 1) % 8
            # Se estiver no mínimo da direção, troca ela
            if pos[1] == self.min[1] and self.direction == 4:
                self.direction = (self.direction + 4) % 8
            if self.obstacles[pos[0],pos[1]][self.dir_view] == VS.END and self.direction == 0:
                if self.dir_view >= 0 and self.dir_view <= 1:
                    self.direction = (self.direction + 4) % 8
                    self.dir_view = 2
            dx, dy = Explorer.AC_INCR[self.dir_view]
            if pos[0] + dx < self.min[0] or pos[1] + dy > self.min[1]:
                return
        
        elif self.quadrante == 1:
            if self.direction == 2:
                self.dir_view = (self.dir_view + 1) % 8
            else:
                self.dir_view = (self.dir_view - 1) % 8
            if pos[0] == self.min[0] and self.direction == 6:
                self.direction = (self.direction + 4) % 8
            if self.obstacles[pos[0],pos[1]][self.dir_view] == VS.END and self.direction == 2:
                if self.dir_view >= 2 and self.dir_view <= 3:
                    self.direction = (self.direction + 4) % 8
                    self.dir_view = 4
            dx, dy = Explorer.AC_INCR[self.dir_view]
            if pos[0] + dx < self.min[0] or pos[1] + dy < self.min[1]:
                return
        
        elif self.quadrante == 2:
            if self.direction == 4:
                self.dir_view = (self.dir_view + 1) % 8
            else:
                self.dir_view = (self.dir_view - 1) % 8
            if pos[1] == self.min[1] and self.direction == 0:
                self.direction = (self.direction + 4) % 8
            if self.obstacles[pos[0],pos[1]][self.dir_view] == VS.END and self.direction == 4:
                if self.dir_view >= 4 and self.dir_view <= 5:
                    self.direction = (self.direction + 4) % 8
                    self.dir_view = 6
            dx, dy = Explorer.AC_INCR[self.dir_view]
            if pos[0] + dx > self.min[0] or pos[1] + dy < self.min[1]:
                return
        
        else:
            if self.direction == 6:
                self.dir_view = (self.dir_view + 1) % 8
            else:
                self.dir_view = (self.dir_view - 1) % 8
            if pos[0] == self.min[0] and self.direction == 2:
                self.direction = (self.direction + 4) % 8
            if self.obstacles[pos[0],pos[1]][self.dir_view] == VS.END and self.direction == 6:
                if self.dir_view >= 6 and self.dir_view <= 7:
                    self.direction = (self.direction + 4) % 8
                    self.dir_view = 0
            dx, dy = Explorer.AC_INCR[self.dir_view]
            if pos[0] + dx > self.min[0] or pos[1] + dy > self.min[1]:
                return
            
        # Verifica se a posição já foi visitada
        if ((pos[0] + dx, pos[1] + dy)) in self.lista:
            return

        # Verifica se o movimento é possível
        if self.obstacles[pos[0],pos[1]][self.dir_view] == VS.CLEAR:
            self.lista.append((pos[0] + dx, pos[1] + dy))
            self.goto_astar((pos[0], pos[1]))
            self.goto.append((dx, dy))
            return self.get_next_position()

    def goto_astar(self, goal):
        start = self.x, self.y
        astar = self.aStar.search(start, goal)

        cmp = 0
        if len(astar) <= 1:
            return
        else:
            cmp = astar[0][0][0], astar[0][0][1]
        for a in astar[1:]:
            x = a[0][0] - cmp[0]
            y = a[0][1] - cmp[1]
            self.goto.append((x,y))
            cmp = a[0]

    def aumenta_quadrante(self):
        # Utiliza o restante da bateria como parametro para aumentar o tamanho do quadrante
        # Assim, o agente tenta explorar mais depois de terminar o quadrante (ou a área que acreditava ser o quadrante)
        print("Aumentando quadrante para ", self.get_name())
        battery = round(math.sqrt(self.get_rtime()))
        if self.quadrante == 0:
            self.min = self.min[0] - battery, self.min[1] + battery
        if self.quadrante == 1:
            self.min = self.min[0] - battery, self.min[1] - battery
        if self.quadrante == 2:
            self.min = self.min[0] + battery, self.min[1] - battery
        if self.quadrante == 3:
            self.min = self.min[0] + battery, self.min[1] + battery
        self.finishedQ = True
        return self.get_next_position()
        
    def explore(self):
        # get an random increment for x and y       
        result = self.get_next_position()
        dx, dy = result

        # Moves the body to another position  
        rtime_bef = self.get_rtime()
        result = self.walk(dx, dy)
        rtime_aft = self.get_rtime()
        
        # Test the result of the walk action
        # Should never bump, but for safe functionning let's test
        if result == VS.BUMPED:
            # update the map with the wall
            self.map.add((self.x + dx, self.y + dy), VS.OBST_WALL, VS.NO_VICTIM, self.check_walls_and_lim())
            #print(f"{self.NAME}: Wall or grid limit reached at ({self.x + dx}, {self.y + dy})")

        if result == VS.EXECUTED:
            # check for victim returns -1 if there is no victim or the sequential
            # the sequential number of a found victim
            self.walk_stack.append((dx, dy))

            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy

            # update the walk time
            self.walk_time = self.walk_time + (rtime_bef - rtime_aft)
            #print(f"{self.NAME} walk time: {self.walk_time}")

            # Check for victims
            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                #print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")
                #print(f"{self.NAME} Seq: {seq} Vital signals: {vs}")
            
            # Calculates the difficulty of the visited cell
            difficulty = (rtime_bef - rtime_aft)
            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            # Update the map with the new cell
            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
            #print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")

            # Update the maximum difficulty
            if difficulty > self.max_Difficulty:
                self.max_Difficulty = difficulty

            # print(f"{self.NAME}:at ({self.x}, {self.y}), diffic: {difficulty:.2f} vict: {seq} rtime: {self.get_rtime()}")
            # Atualiza tempo de retorno
            self.aStar.set_difficulty((self.x, self.y), difficulty)
            # self.return_time = self.comeBack.heuristic((self.x, self.y), difficulty)
            
            # print(f"Max difficulty: {self.max_Difficulty}")

        return

    def come_back(self, back = (0,0)):
        if self.last_path:
            self.return_path = self.aStar.search((self.x, self.y), back)
            self.last_path = False

        # The first element is the (self.x, self.y), the second is the next position
        nextPosition = self.return_path[self.it_return_path][0]
        
        if self.it_return_path <= len(self.return_path) - 1:
            self.it_return_path += 1
        else:
            self.it_return_path = 1
            self.last_path = True

        dx = nextPosition[0] - self.x
        dy = nextPosition[1] - self.y
        
        result = self.walk(dx, dy)
        
        if result == VS.BUMPED:
            print(f"{self.NAME}: when coming back bumped at ({self.x+dx}, {self.y+dy}) , rtime: {self.get_rtime()}")
            
        elif result == VS.EXECUTED:
            # update the agent's position relative to the origin
            self.x += dx
            self.y += dy
            
        
    def deliberate(self) -> bool:
        """ The agent chooses the next action. The simulator calls this
        method at each cycle. Must be implemented in every agent"""

        # forth and back: go, read the vital signals and come back to the position
        
        if self.update_cont >= 200:
            self.return_path = self.aStar.search((self.x, self.y), (0, 0))
                
            last_item = self.return_path[::-1][0]
            
            self.return_time = last_item[1]
            
            self.update_cont = 0
        else:
            self.update_cont += 1
        
        # keeps exploring while there is enough time
        if self.get_rtime() > self.return_time * 2 + 50 and not(self.finishedAll):
            self.explore()
            return True

        # no more come back walk actions to execute or already at base
        if (self.x == 0 and self.y == 0): #or self.walk_stack.is_empty():
            # time to pass the map and found victims to the master rescuer
            self.resc.sync_explorers(self.map, self.victims)
            # finishes the execution of this agent
            return False
        
        # proceed to the base
        self.come_back()
        return True

