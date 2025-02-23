# algoritmo_genetico.py
import numpy as np
from typing import List
from pqa import PQA
from operadores import Selecao, Crossover, Mutacao, Elitismo

class AlgoritmoGenetico:
    def __init__(self, pqc: PQA, tamanho_populacao: int, max_geracoes: int, 
                 taxa_mutacao: float, taxa_elitismo: float,
                 metodo_selecao: str = 'torneio',
                 metodo_crossover: str = 'ox',
                 metodo_elitismo: str = 'top',
                 metodo_mutacao: str = 'swap'):
        
        self.pqc = pqc
        self.tamanho_populacao = tamanho_populacao
        self.max_geracoes = max_geracoes
        self.taxa_mutacao = taxa_mutacao
        self.taxa_elitismo = taxa_elitismo
        self.metodo_selecao = metodo_selecao
        self.metodo_crossover = metodo_crossover
        self.metodo_elitismo = metodo_elitismo
        self.metodo_mutacao = metodo_mutacao
        self.populacao = self._gerar_populacao_inicial()

    def _gerar_populacao_inicial(self) -> List[List[int]]:
        return [np.random.permutation(self.pqc.n).tolist() 
                for _ in range(self.tamanho_populacao)]

    def _calcular_fitness(self) -> List[float]:
        return [1 / (1 + self.pqc.calcular_custo(ind)) for ind in self.populacao]

    def executar(self) -> List[int]:
        melhor_custo_historico = []
        
        for geracao in range(self.max_geracoes):
            fitness = self._calcular_fitness()
            melhor_custo = int(1 / max(fitness) - 1)
            media_custo = int(1 / np.mean(fitness) - 1)
            pior_custo = int(1 / min(fitness) - 1)
            
            melhor_custo_historico.append(melhor_custo)
            
            print(f"Geração {geracao:3d} | "
                  f"Melhor: {melhor_custo:5d} | "
                  f"Média: {media_custo:5d} | "
                  f"Pior: {pior_custo:5d}")

            # Seleção
            if self.metodo_selecao == 'torneio':
                selecionados = Selecao.torneio_eficiente(self.populacao, fitness)
            elif self.metodo_selecao == 'roleta':
                selecionados = Selecao.roleta(self.populacao, fitness)

            # Crossover
            filhos = []
            n_elites = max(1, int(self.taxa_elitismo * self.tamanho_populacao))
            
            for _ in range(self.tamanho_populacao - n_elites):
                pai1, pai2 = selecionados[np.random.randint(len(selecionados))], \
                            selecionados[np.random.randint(len(selecionados))]
                
                if self.metodo_crossover == 'ox':
                    filho = Crossover.ox(pai1, pai2)
                elif self.metodo_crossover == 'pmx':
                    filho = Crossover.pmx(pai1, pai2)
                
                # Mutação
                if np.random.rand() < self.taxa_mutacao:
                    if self.metodo_mutacao == 'swap':
                        filho = Mutacao.swap(filho)
                    elif self.metodo_mutacao == 'inversao':
                        filho = Mutacao.inversao(filho)
                
                filhos.append(filho)

            # Elitismo
            if self.metodo_elitismo == 'top':
                elites = Elitismo.manter_melhores(self.populacao, fitness, n_elites)
            elif self.metodo_elitismo == 'hibrido':
                n_aleatorios = max(1, int(n_elites * 0.5))
                elites = Elitismo.manter_melhores_e_aleatorios(
                    self.populacao, fitness, n_elites, n_aleatorios)
            
            self.populacao = elites + filhos

            # Critério de parada
            if geracao >= 50:
                diferenca = abs(melhor_custo_historico[-50] - melhor_custo)
                if diferenca / melhor_custo_historico[-50] < 0.001:
                    print(f"Convergência na geração {geracao}!")
                    break

        return self.populacao[np.argmax(self._calcular_fitness())]
