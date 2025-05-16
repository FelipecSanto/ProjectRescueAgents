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

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        else:
            return None

    def is_empty(self):
        return len(self.items) == 0

class Node:
    def __init__(self, data):
        self.data = data  # data will be of the form [x, y]
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def add(self, item):
        """Add a new item to the end of the list."""
        new_node = Node(item)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node

    def remove(self, item):
        """Remove the first occurrence of the item from the list."""
        current = self.head
        prev = None
        while current:
            if current.data == item:
                if prev:
                    prev.next = current.next
                else:
                    self.head = current.next
                return
            prev = current
            current = current.next

    def find(self, item):
        """Find an item in the list. Returns True if found, False otherwise."""
        current = self.head
        while current:
            if current.data == item:
                return True
            current = current.next
        return False

    def get(self, index):
        """Get the item at a specific index. Returns None if index is out of bounds."""
        current = self.head
        count = 0
        while current:
            if count == index:
                return current.data
            current = current.next
            count += 1
        return None

    def size(self):
        """Return the size of the list."""
        count = 0
        current = self.head
        while current:
            count += 1
            current = current.next
        return count

    def is_empty(self):
        """Check if the list is empty."""
        return self.head is None

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
        self.walk_stack = Stack()  # a stack to store the movements
        self.walk_time = 0        # time consumed to walk when exploring (to decide when to come back)
        self.set_state(VS.ACTIVE)  # explorer is active since the begin
        self.resc = resc           # reference to the rescuer agent
        self.x = 0                 # current x position relative to the origin 0
        self.y = 0                 # current y position relative to the origin 0
        self.map = Map()           # create a map for representing the environment
        self.victims = {}          # a dictionary of found victims: (seq): ((x,y), [<vs>])
                                   # the key is the seq number of the victim,(x,y) the position, <vs> the list of vital signals
        self.quadrante = dir - 1
        self.direction = (dir - 1) * 2
        self.lista = LinkedList()
        self.return_time = 0
        
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
        # Checa obstáculos em volta
        obstacles = self.check_walls_and_lim()

        dir = (self.direction + 4) % 8

        cont = 0
        dx, dy = 0, 0

        while True:
            cont += 1
            if cont > 8:
                result = self.walk_stack.pop()
                if result is None:
                    return (0, 0), 1
                dx, dy = result
                dx = dx * -1
                dy = dy * -1
                result = dx, dy
                return result, 1
            if self.quadrante == 0:
                if self.direction == 0:
                    dir = (dir + 1) % 8
                else:
                    dir = (dir - 1) % 8
                if self.y == 0 and self.direction == 4:
                    self.direction = (self.direction + 4) % 8
                    dir = 2
                if obstacles[dir] == VS.END and self.direction == 0:    
                    self.direction = (self.direction + 4) % 8
                    dir = 2
                dx, dy = Explorer.AC_INCR[dir]
                if self.x + dx < 0 or self.y + dy > 0:
                    continue
            if self.quadrante == 1:
                if self.direction == 2:
                    dir = (dir + 1) % 8
                else:
                    dir = (dir - 1) % 8
                if self.x == 0 and self.direction == 6:
                    self.direction = (self.direction + 4) % 8
                    dir = 4
                if obstacles[dir] == VS.END and self.direction == 2:
                    self.direction = (self.direction + 4) % 8
                    dir = 4
                dx, dy = Explorer.AC_INCR[dir]
                if self.x + dx < 0 or self.y + dy < 0:
                    continue
            if self.quadrante == 2:
                if self.direction == 4:
                    dir = (dir + 1) % 8
                else:
                    dir = (dir - 1) % 8
                if self.x == 0 and self.direction == 0:
                    self.direction = (self.direction + 4) % 8
                    dir = 6
                if obstacles[dir] == VS.END and self.direction == 4:
                    self.direction = (self.direction + 4) % 8
                    dir = 6
                dx, dy = Explorer.AC_INCR[dir]
                if self.x + dx > 0 or self.y + dy < 0:
                    continue
            if self.quadrante == 3:
                if self.direction == 6:
                    dir = (dir + 1) % 8
                else:
                    dir = (dir - 1) % 8
                if self.y == 0 and self.direction == 2:
                    self.direction = (self.direction + 4) % 8
                    dir = 0
                if obstacles[dir] == VS.END and self.direction == 6:
                    self.direction = (self.direction + 4) % 8
                    dir = 0
                dx, dy = Explorer.AC_INCR[dir]
                if self.x + dx > 0 or self.y + dy > 0:
                    continue
            if self.lista.find([self.x + dx, self.y + dy]) is True:
                continue
            if obstacles[dir] == VS.CLEAR:
                self.lista.add([self.x + dx, self.y + dy])
                return Explorer.AC_INCR[dir], 0
                
        
    def explore(self):
        # get an random increment for x and y       
        result = self.get_next_position()
        dx, dy = result[0]
        back = result[1]

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
            if back == 0:
                self.walk_stack.push((dx, dy))

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

    def come_back(self):
        dx, dy = self.walk_stack.pop()
        dx = dx * -1
        dy = dy * -1

        # if self.NAME == "EXPL_1":
        #     print(f"{self.NAME}: going back to the base, rtime: {self.get_rtime()}")

        if self.last_path:
            self.return_path = self.aStar.search((self.x, self.y), (0, 0))
            self.last_path = False

        # The first element is the (self.x, self.y), the second is the next position
        nextPosition = self.return_path[self.it_return_path][0]
        
        self.it_return_path += 1

        dx = nextPosition[0] - self.x
        dy = nextPosition[1] - self.y

        # if self.NAME == "EXPL_1":
        #     print(f"{self.NAME}: walking to the base, going to ({self.x+dx}, {self.y+dy}), rtime: {self.get_rtime()}")
        
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
        if self.get_rtime() > self.return_time * 2 + 50:
            self.explore()
            return True

        # no more come back walk actions to execute or already at base
        if (self.x == 0 and self.y == 0) or self.walk_stack.is_empty():
            # time to pass the map and found victims to the master rescuer
            self.resc.sync_explorers(self.map, self.victims)
            # finishes the execution of this agent
            return False
        
        # proceed to the base
        self.come_back()
        return True

