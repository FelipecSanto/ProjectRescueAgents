VictimSim2
==========

Um simulador desenvolvido para testar algoritmos de busca e outras técnicas de Inteligência Artificial, utilizado na disciplina de Inteligência Artificial na UTFPR, campus Curitiba. Conhecido como VictimSim2, este simulador é útil para o estudo de cenários catastróficos em um ambiente 2D em grade, onde agentes artificiais realizam missões de busca e salvamento para localizar e ajudar vítimas.

Principais características do simulador
---------------------------------------

- O ambiente é composto por uma grade 2D, indexada por coordenadas (coluna, linha) ou (x, y). A origem está situada no canto superior esquerdo, com o eixo y se estendendo para baixo e o eixo x para a direita. Embora as coordenadas absolutas estejam acessíveis apenas ao simulador de ambiente, os usuários são encorajados a definir seu próprio sistema de coordenadas para os agentes.
- Cada célula da grade 2D possui um grau de dificuldade para acessibilidade, com valores que vão de maiores que zero até 100. O valor máximo de 100 indica a presença de uma parede intransponível, enquanto valores mais altos indicam acesso cada vez mais difícil. Por outro lado, valores menores ou iguais a 1 indicam entrada mais fácil.
- O ambiente permite um ou mais agentes, sendo possível personalizar a cor de cada agente por meio de arquivos de configuração.
- O sistema possui detecção de colisões para identificar quando um agente colide com paredes ou atinge os limites da grade, percepção conhecida como "BUMPED".
- Os agentes têm a capacidade de detectar obstáculos e os limites da grade no seu entorno imediato, ou seja, a um passo de sua posição atual.
- Vários agentes podem ocupar a mesma célula simultaneamente sem causar colisões.
- O simulador gerencia o agendamento de cada agente com base em seu estado: ACTIVE, IDLE, ENDED ou DEAD. Apenas agentes ativos podem executar ações, e o simulador controla o tempo de execução permitido para cada agente; ao término desse tempo, o agente é considerado DEAD.
