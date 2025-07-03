import sys
import os
import time

# Adiciona o diretório raiz do projeto ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Importa classes
from vs.environment import Env
from explorer import Explorer
from rescuer import Rescuer

# Importa funções de clustering
from clustering import cluster_victims, save_clusters


def main(data_folder_name, config_ag_folder_name):
    # Set the path to config files and data files for the environment
    current_folder = os.path.abspath(os.getcwd())
    config_ag_folder = os.path.abspath(os.path.join(current_folder, config_ag_folder_name))
    data_folder = os.path.abspath(os.path.join(current_folder, data_folder_name))

    # Instancia o ambiente
    env = Env(data_folder)

    # Instancia o agente mestre socorrista (vai receber os mapas)
    rescuer_file = os.path.join(config_ag_folder, "rescuer_1_config.txt")
    master_rescuer = Rescuer(env, rescuer_file, config_ag_folder, 4)  # 4 é o número de exploradores

    # Instancia os exploradores, que conhecem o mestre para sincronizar os dados
    for exp in range(1, 5):
        filename = f"explorer_{exp:1d}_config.txt"
        explorer_file = os.path.join(config_ag_folder, filename)
        Explorer(env, explorer_file, master_rescuer, exp)

    # Executa a simulação no ambiente para a exploração
    env.run()
    
    # Exibe resultados acumulados no terminal
    env.print_results()
    env.print_acum_results()
    


if __name__ == '__main__':
    """Para usar dados de uma pasta diferente, passar via argumento"""
    if len(sys.argv) > 1:
        data_folder_name = sys.argv[1]
    else:
        data_folder_name = os.path.join("datasets", "data_430v_100x100")

    config_ag_folder_name = os.path.join("src", "cfg_1")

    main(data_folder_name, config_ag_folder_name)
