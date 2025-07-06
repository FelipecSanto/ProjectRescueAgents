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
from Astar import Astar
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from bfs import BFS
from abc import ABC, abstractmethod
import joblib
import subprocess

# Importa funções de clustering
from clustering import cluster_victims, save_clusters

## Classe que define o Agente Rescuer com um plano fixo
class Rescuer(AbstAgent):
    def __init__(self, env, config_file, config_ag_folder, nb_of_explorers=1,clusters=[]):
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
        self.sequences = []          # the sequence of visit of victims for each cluster 
        self.sequences_left = []
        self.victims_left = []
        
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

        # Salva os clusters usando os valores de severidade e classe previstos
        self.save_clusters_csv(self.clusters)

    def predict_severity_and_class(self):
        """ Prediz gravidade (regressão) e classe (classificação) para cada vítima usando modelos treinados.
        Usa os sinais vitais como entrada e armazena os resultados na lista de sinais vitais da vítima. """
        
        # Caminho dos modelos
        MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "model"))
        rf_reg_path = os.path.join(MODEL_DIR, "best_rf_reg.joblib")
        xgb_clf_path = os.path.join(MODEL_DIR, "best_xgb_clf.joblib")
        scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")

        # Tenta carregar os modelos, se não existir, chama o script de treino
        try:
            print("Carregando modelos...")
            rf_reg = joblib.load(rf_reg_path)
            xgb_clf = joblib.load(xgb_clf_path)
            scaler = joblib.load(scaler_path)
        except FileNotFoundError:
            print("Modelos não encontrados. Treinando modelos...")
            # Executa o script de treinamento
            script_path = os.path.join(MODEL_DIR, "train_rescuer_models.py")
            if os.name == "nt":  # Windows
                subprocess.run(["python", script_path], check=True)
            else:  # POSIX (Linux, macOS, etc.)
                subprocess.run(["python3", script_path], check=True)
            # Tenta carregar novamente
            rf_reg = joblib.load(rf_reg_path)
            xgb_clf = joblib.load(xgb_clf_path)
            scaler = joblib.load(scaler_path)
        
        for vic_id, values in self.victims.items():
            coords, vs = values

            # Garante que estamos usando apenas os sinais vitais relevantes (6 primeiros)
            if len(vs) < 6:
                continue  # ignora vítima com dados incompletos

            vitals = vs[:6]
            qPA, pulso, freq_resp = vitals[0], vitals[1], vitals[2]

            # Engenharia de atributos como no treino
            qPA_pulso_ratio = qPA / pulso if pulso != 0 else 0
            pulso_freq_prod = pulso * freq_resp
            qPA_minus_pulso = qPA - pulso

            features = [qPA, pulso, freq_resp, qPA_pulso_ratio, pulso_freq_prod, qPA_minus_pulso]
            features_scaled = scaler.transform([features])

            grav_pred = rf_reg.predict(features_scaled)[0]
            classe_pred = xgb_clf.predict(features_scaled)[0]

            # Atualiza sinais vitais com as predições
            vs = vs[:6]  # remove predições anteriores
            vs.extend([grav_pred, int(classe_pred) + 1])  # + 1 Corrige classe para 1-4
            self.victims[vic_id] = (coords, vs)


    def sequencing(self, cluster, left = []):
        """
        Sequenciamento das vítimas
        """

        # Posição inicial
        pos_atual = self.x, self.y

        # Cópia das vítimas do cluster
        if not left:
            victims = set(cluster)
        else:
            victims = set(left)
            
        sequence = []

        while victims:
            # Encontra a vítima mais próxima da posição atual com um peso da classe
            close_victim = min(
                victims,
                key=lambda v: (((self.victims[v][0][0] - pos_atual[0])**2 + 
                               (self.victims[v][0][1] - pos_atual[1])**2)
                               *
                                (self.victims[v][1][7] * 1.1))
            )

            # Adiciona à sequência
            sequence.append(close_victim)

            # Atualiza a posição atual
            pos_atual = (
                self.victims[close_victim][0][0],
                self.victims[close_victim][0][1]
            )

            # Remove a vítima visitada
            victims.remove(close_victim)

        if not left:
            self.sequences = sequence
        else:
            self.sequences_left = sequence

    def planner(self, total_left = []):
        """ A method that calculates the path between victims: walk actions in a OFF-LINE MANNER (the agent plans, stores the plan, and
            after it executes. Eeach element of the plan is a pair dx, dy that defines the increments for the the x-axis and  y-axis."""

        aStar = Astar(self.map, self)

        # for each victim of the first sequence of rescue for this agent, we're going go calculate a path
        # starting at the base - always at (0,0) in relative coords
        
        if not self.sequences:   # no sequence assigned to the agent, nothing to do
            return

        # we consider only the first sequence (the simpler case)
        # The victims are sorted by x followed by y positions: [vic_id]: ((x,y), [<vs>]

        base = (0,0)
        start = (0,0)
        total_plan = []

        sequence = []
        if not total_left:
            sequence = self.sequences
        else:
            sequence = self.sequences_left

        for vic_id in sequence:
            goal = self.victims[vic_id][0]
            plan = aStar.search(start, goal)
            time = plan[len(plan)- 1][1] * 1.2 + 1 # Assume que pode perder mais tempo do que o planejado
            base_plan = aStar.search(goal, base) 
            base_time = base_plan[len(base_plan) - 1][1] * 1.2 # O mesmo vale para o caminho de volta ao base
            if(self.plan_rtime - time < base_time + 60): # +60 de gap
                self.victims_left.append(vic_id)
                continue

            total_plan = total_plan + plan
            self.plan_rtime = self.plan_rtime - time
            start = goal

            if vic_id in total_left:
                total_left.remove(vic_id)

        # Plan to come back to the base
        plan = aStar.search(start, base)
        total_plan = total_plan + plan
        if start != base:
            anterior = total_plan[1][0]
        else:
            anterior = total_plan[0][0]
        self.plan.append(anterior) # add the first action to the plan
        for p in total_plan[2:]:
            x = p[0][0] - anterior[0]
            y = p[0][1] - anterior[1]
            self.plan.append((x,y))
            anterior = p[0]

    def sync_explorers(self, explorer_map, victims):
        # Atualiza mapa global
        self.map.update(explorer_map)
        # Atualiza mapa global de vítimas
        for vid, (coords, signals) in victims.items():
            self.victims[vid] = (coords, signals)
            
        self.received_maps += 1
        
        if self.received_maps == self.nb_of_explorers:
            print("Fase de exploração terminada")
            rescuers = []
            rescuers.append(self)

            self.predict_severity_and_class()
            self.cluster_victims()
            self.sequencing(self.clusters[0])  
            self.planner()
            self.set_state(VS.ACTIVE)  # set the rescuer to ACTIVE state

            for rescuer in range(2, self.nb_of_explorers + 1):
                rescuers.append(self.setup_rescuer(rescuer))

            # Junta todos as vítimas restantes que não foram salvas pelo rescuer do cluster
            total_left = []
            for r in rescuers:
                total_left = total_left + r.victims_left

            # Joga as vítimas restantes no rescuer que tem tempo (No caso, que não tem nenhuma vítima restante)
            for r in rescuers:
                if not r.victims_left and total_left:
                    r.sequencing(r.sequences, total_left)
                    r.planner(total_left)

    def setup_rescuer(self, index):
        filename = f"rescuer_{index}_config.txt"
        rescuer_file = os.path.join(self.config_ag_folder, filename)
        rescuer = Rescuer(self.env, rescuer_file, self.config_ag_folder, self.nb_of_explorers)

        # Compartilha os dados do mestre com os demais
        rescuer.victims = self.victims
        rescuer.clusters = self.clusters
        rescuer.map = self.map

        rescuer.sequencing(self.clusters[index-1])
        rescuer.planner()
        rescuer.set_state(VS.ACTIVE)
        return rescuer
    
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
                    self.plan_rtime -= 1
                    #if self.first_aid(): # True when rescued
                        #print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")                    
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.x})")
            
        return True

