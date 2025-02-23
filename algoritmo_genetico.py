import numpy as np
from typing import List
from pqa import PQA
from operadores import Selecao, Crossover, Mutacao, Elitismo

class AlgoritmoGenetico:
    def __init__(self, pqc: PQA, tamanho_populacao: int, max_geracoes: int, 
                 taxa_mutacao: float, taxa_elitismo: float):
        self.pqc = pqc
        self.tamanho_populacao = tamanho_populacao
        self.max_geracoes = max_geracoes
        self.taxa_mutacao = taxa_mutacao
        self.taxa_elitismo = taxa_elitismo
        self.populacao = self._gerar_populacao_inicial()

    def _gerar_populacao_inicial(self) -> List[List[int]]:
        return [np.random.permutation(self.pqc.n).tolist() 
                for _ in range(self.tamanho_populacao)]

    def _calcular_fitness(self) -> List[float]:
        return [1 / (1 + self.pqc.calcular_custo(ind)) for ind in self.populacao]

    def executar(self) -> List[int]:
        melhor_custo_historico = []
        n_elites = max(1, int(self.taxa_elitismo * self.tamanho_populacao))
        
        for geracao in range(self.max_geracoes):
            fitness = self._calcular_fitness()
            melhor_custo = int(1 / max(fitness) - 1)  # Converte fitness para custo
            media_custo = int(1 / np.mean(fitness) - 1)
            pior_custo = int(1 / min(fitness) - 1)
            
            melhor_custo_historico.append(melhor_custo)
            
            print(f"Geração {geracao:3d} | "
                  f"Melhor: {melhor_custo:5d} | "
                  f"Média: {media_custo:5d} | "
                  f"Pior: {pior_custo:5d}")    
            
            # Seleção por torneio
            selecionados = Selecao.torneio_eficiente(self.populacao, fitness)
            
            # Crossover e Mutação
            filhos = []
            for _ in range(self.tamanho_populacao - n_elites):
                idx_pai1, idx_pai2 = np.random.choice(len(selecionados), size=2, replace=True)
                pai1 = selecionados[idx_pai1]
                pai2 = selecionados[idx_pai2]
                filho = Crossover.ox(pai1, pai2)
                if np.random.rand() < self.taxa_mutacao:
                    Mutacao.swap(filho)
                filhos.append(filho)
            
            # Elitismo e Nova População
            elites = Elitismo.manter_melhores(self.populacao, fitness, n_elites)
            self.populacao = elites + filhos[:self.tamanho_populacao - n_elites]

            # Critério de parada aprimorado
            if geracao >= 20:
                diferenca_percentual = abs(melhor_custo_historico[-20] - melhor_custo) / melhor_custo_historico[-20] * 100
                if diferenca_percentual < 0.1:  # 0.1% de melhoria em 20 gerações
                    print(f"Convergência na geração {geracao} (melhoria menor que 0.1% nos últimos 20 gerações)!")
                    break

        return self.populacao[np.argmax(self._calcular_fitness())]
