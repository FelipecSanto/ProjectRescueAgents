##  RESCUER AGENT
### @Author: Tacla (UTFPR)
### Demo of use of VictimSim
### This rescuer version implements:
### - clustering of victims by quadrants of the explored region 
### - definition of a sequence of rescue of victims of a cluster
### - assigning one cluster to one rescuer
### - calculating paths between pair of victims using breadth-first search
###
### One of the rescuers is the master in charge of unifying the maps and the information
### about the found victims.

import os
import random
import math
import csv
import sys
import numpy as np
from sklearn.cluster import KMeans
from map import Map
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from bfs import BFS
from abc import ABC, abstractmethod

# Importa funções de clustering
from clustering import cluster_victims, save_clusters

## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    def __init__(self, env, config_file, config_ag_folder, nb_of_explorers=1,clusters=[],):
        """ 
        @param env: a reference to an instance of the environment class
        @param config_file: the absolute path to the agent's config file
        @param nb_of_explorers: number of explorer agents to wait for
        @param clusters: list of clusters of victims in the charge of this agent"""

        super().__init__(env, config_file)

        # Specific initialization for the rescuer
        self.nb_of_explorers = nb_of_explorers       # number of explorer agents to wait for start
        self.received_maps = 0                       # counts the number of explorers' maps
        self.map = Map()                             # explorer will pass the map
        self.victims = {}            # a dictionary of found victims: [vic_id]: ((x,y), [<vs>])
        self.plan = []               # a list of planned actions in increments of x and y
        self.plan_x = 0              # the x position of the rescuer during the planning phase
        self.plan_y = 0              # the y position of the rescuer during the planning phase
        self.plan_visited = set()    # positions already planned to be visited 
        self.plan_rtime = self.TLIM  # the remaing time during the planning phase
        self.plan_walk_time = 0.0    # previewed time to walk during rescue
        self.x = 0                   # the current x position of the rescuer when executing the plan
        self.y = 0                   # the current y position of the rescuer when executing the plan
        self.clusters = clusters     # the clusters of victims this agent should take care of - see the method cluster_victims
        self.sequences = clusters    # the sequence of visit of victims for each cluster 
        
        self.config_ag_folder = config_ag_folder
        self.env = env
        
                
        # Starts in IDLE state.
        # It changes to ACTIVE when the map arrives
        self.set_state(VS.IDLE)

    def save_clusters_csv(self, clusters):
        """
        Salva os clusters em arquivos, incluindo severidade e classe previstos.
        """
        print("Salvando clusters em /clusters...")

        # Caminho absoluto para o diretório pai da pasta atual (mas)
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        clusters_dir = os.path.join(parent_dir, "clusters")
        os.makedirs(clusters_dir, exist_ok=True)

        for i, cluster in enumerate(clusters, start=1):
            file_path = os.path.join(clusters_dir, f"cluster{i}.txt")
            with open(file_path, "w") as f:
                for vid in cluster:
                    x, y = self.victims[vid][0]
                    vs = self.victims[vid][1]
                    # Usa os valores previstos de severidade e classe (índices -2 e -1)
                    grav = ""
                    classe = ""
                    if len(vs) >= 8:
                        grav = vs[-2]
                        classe = vs[-1]
                    f.write(f"{vid},{x},{y},{grav},{classe}\n")

        print("Clusters salvos em /clusters")

    def save_sequence_csv(self, sequence, sequence_id):
        
        # Caminho absoluto para o diretório pai da pasta atual (mas)
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        seqs_dir = os.path.join(parent_dir, "seqs")
        os.makedirs(seqs_dir, exist_ok=True)
        
        filename = os.path.join(seqs_dir, f"seq{sequence_id}.txt")
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for id, values in sequence.items():
                x, y = values[0]      # x,y coordinates
                vs = values[1]        # list of vital signals
                writer.writerow([id, x, y, vs[6], vs[7]])

    def cluster_victims(self):
        """ Agrupa vítimas em clusters com base em suas coordenadas (x, y).

                :param victims: dicionário no formato {id: ((x, y), sinais_vitais)}
                :param n_clusters: número de clusters (igual ao número de socorristas)
                
                :return: lista de clusters, cada um é uma lista de ids de vítimas
        """
        
        print("Agrupando vítimas em", self.nb_of_explorers, "clusters...")

        if len(self.victims) < self.nb_of_explorers:
            raise ValueError(f"Número de vítimas ({len(self.victims)}) é menor que o número de clusters ({self.nb_of_explorers})")

        victim_ids = list(self.victims.keys())
        coords = np.array([self.victims[vid][0] for vid in victim_ids])  # extrai (x,y) de cada vítima

        # Aplica KMeans para gerar os clusters
        kmeans = KMeans(n_clusters=self.nb_of_explorers, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(coords)

        # Agrupa os ids por rótulo de cluster
        clusters = {i: [] for i in range(self.nb_of_explorers)}
        for vid, label in zip(victim_ids, labels):
            clusters[label].append(vid)

        print("Clusters criados!")

        self.clusters = list(clusters.values())
        self.sequences = self.clusters

        # Salva os clusters usando os valores de severidade e classe previstos
        self.save_clusters_csv(self.clusters)

    def predict_severity_and_class(self):
        """
        Prediz a gravidade (valor contínuo) e a classe (1 a 4) para cada vítima.
        Neste exemplo, usamos um modelo fictício: a gravidade é uma função dos sinais vitais,
        e a classe é baseada em limiares desse valor.
        """
        for vic_id, values in self.victims.items():
            vs = values[1]
            # Exemplo: gravidade como média dos sinais vitais (ajuste conforme necessário)
            severity_value = float(np.mean(vs[:6]))  # supondo que os 6 primeiros são sinais vitais
            # Classificação baseada em limiares arbitrários
            if severity_value > 75:
                severity_class = 1  # Crítico
            elif severity_value > 50:
                severity_class = 2  # Instável
            elif severity_value > 25:
                severity_class = 3  # Potencialmente estável
            else:
                severity_class = 4  # Estável
            # Adiciona ao vetor de sinais vitais
            # Remove valores antigos se já existirem (evita duplicação)
            if len(vs) > 6:
                vs = vs[:6]
            vs.extend([severity_value, severity_class])
            self.victims[vic_id] = (values[0], vs)


    def sequencing(self):
        """ Currently, this method sort the victims by the x coordinate followed by the y coordinate
            @TODO It must be replaced by a Genetic Algorithm that finds the possibly best visiting order """

        """ We consider an agent may have different sequences of rescue. The idea is the rescuer can execute
            sequence[0], sequence[1], ...
            A sequence is a dictionary with the following structure: [vic_id]: ((x,y), [<vs>]"""

        new_sequences = []

        for seq in self.sequences:   # a list of sequences, being each sequence a dictionary
            seq = dict(sorted(seq.items(), key=lambda item: item[1]))
            new_sequences.append(seq)       
            #print(f"{self.NAME} sequence of visit:\n{seq}\n")

        self.sequences = new_sequences

    def planner(self):
        """ A method that calculates the path between victims: walk actions in a OFF-LINE MANNER (the agent plans, stores the plan, and
            after it executes. Eeach element of the plan is a pair dx, dy that defines the increments for the the x-axis and  y-axis."""


        # let's instantiate the breadth-first search
        bfs = BFS(self.map, self.COST_LINE, self.COST_DIAG)

        # for each victim of the first sequence of rescue for this agent, we're going go calculate a path
        # starting at the base - always at (0,0) in relative coords
        
        if not self.sequences:   # no sequence assigned to the agent, nothing to do
            return

        # we consider only the first sequence (the simpler case)
        # The victims are sorted by x followed by y positions: [vic_id]: ((x,y), [<vs>]

        sequence = self.sequences[0]
        start = (0,0) # always from starting at the base
        for vic_id in sequence:
            goal = sequence[vic_id][0]
            plan, time = bfs.search(start, goal, self.plan_rtime)
            self.plan = self.plan + plan
            self.plan_rtime = self.plan_rtime - time
            start = goal

        # Plan to come back to the base
        goal = (0,0)
        plan, time = bfs.search(start, goal, self.plan_rtime)
        self.plan = self.plan + plan
        self.plan_rtime = self.plan_rtime - time

    def sync_explorers(self, explorer_map, victims):
        # Atualiza mapa global
        self.map.update(explorer_map)
        # Atualiza mapa global de vítimas
        for vid, (coords, signals) in victims.items():
            self.victims[vid] = (coords, signals)
            
        self.received_maps += 1
        
        if self.received_maps == self.nb_of_explorers:
            print("Fase de exploração terminada")
            self.predict_severity_and_class()
            self.cluster_victims()
            for exp in range(2, self.nb_of_explorers + 1):
                filename = f"rescuer_{exp:1d}_config.txt"
                rescuer_file = os.path.join(self.config_ag_folder, filename)
                Rescuer(self.env, rescuer_file, self.config_ag_folder, self.nb_of_explorers, self.clusters)
        
    def deliberate(self) -> bool:
        """ This is the choice of the next action. The simulator calls this
        method at each reasonning cycle if the agent is ACTIVE.
        Must be implemented in every agent
        @return True: there's one or more actions to do
        @return False: there's no more action to do """

        # No more actions to do
        if self.plan == []:  # empty list, no more actions to do
           print(f"{self.NAME} has finished the plan [ENTER]")
           return False

        # Takes the first action of the plan (walk action) and removes it from the plan
        dx, dy = self.plan.pop(0)
        #print(f"{self.NAME} pop dx: {dx} dy: {dy} ")

        # Walk - just one step per deliberation
        walked = self.walk(dx, dy)

        # Rescue the victim at the current position
        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            #print(f"{self.NAME} Walk ok - Rescuer at position ({self.x}, {self.y})")

            # check if there is a victim at the current position
            if self.map.in_map((self.x, self.y)):
                vic_id = self.map.get_vic_id((self.x, self.y))
                if vic_id != VS.NO_VICTIM:
                    self.first_aid()
                    #if self.first_aid(): # True when rescued
                        #print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")                    
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x})")
            
        return True

