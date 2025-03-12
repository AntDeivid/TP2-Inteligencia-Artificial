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
                 metodo_mutacao: str = 'swap',
                 pmx_retorna_um_filho: bool = False):  # Adicionado
        
        self.pqc = pqc
        self.tamanho_populacao = tamanho_populacao
        self.max_geracoes = max_geracoes
        self.taxa_mutacao = taxa_mutacao
        self.taxa_elitismo = taxa_elitismo
        self.metodo_selecao = metodo_selecao
        self.metodo_crossover = metodo_crossover
        self.metodo_elitismo = metodo_elitismo
        self.metodo_mutacao = metodo_mutacao
        self.pmx_retorna_um_filho = pmx_retorna_um_filho # Salvar o parâmetro
        self.populacao = self._gerar_populacao_inicial()

    def _gerar_populacao_inicial(self) -> List[List[int]]:#vai gerar permutação aletoria ajuda na diversividade
        return [np.random.permutation(self.pqc.n).tolist()
                for _ in range(self.tamanho_populacao)]

    def _calcular_fitness(self) -> List[float]:
        return [1 / (1 + self.pqc.calcular_custo(ind)) for ind in self.populacao]

    def aplicar_crossover(self, pai1: List[int], pai2: List[int]) -> List[int]:
        """Aplica o crossover com base no método selecionado."""
        if self.metodo_crossover == 'ox':
            return Crossover.ox(pai1, pai2)
        elif self.metodo_crossover == 'pmx':
            filho1, filho2 = Crossover.pmx(pai1, pai2)
            if self.pmx_retorna_um_filho:
                return filho1  # Retorna apenas o primeiro filho
            else:
                # Se não for para retornar apenas um filho, retornar um deles aleatoriamente
                # ou levantar um erro dependendo da sua lógica desejada.
                # Por exemplo, pode ser mais correto lançar uma exceção aqui
                # se a flag pmx_retorna_um_filho = False, pois significa que
                # o código chamador não está esperando 2 filhos.
                return np.random.choice([filho1, filho2])
        else:
            raise ValueError(f"Método de crossover desconhecido: {self.metodo_crossover}")

    # algoritmo_genetico.py (trecho corrigido)
    def executar(self) -> List[int]:
        melhor_custo_global = float('inf')
        geracoes_sem_melhoria = 0

        for geracao in range(self.max_geracoes):
            custos = [self.pqc.calcular_custo(ind) for ind in self.populacao]
            fitness = [1 / (1 + custo) for custo in custos]

            melhor_custo = min(custos)
            media_custo = int(np.mean(custos))  # Converta explicitamente para int
            pior_custo = max(custos)

            # Linha corrigida (garanta UTF-8 e formatação correta)
            print(f"Geração {geracao:3d} | Melhor: {melhor_custo:5d} | Média: {media_custo:5d} | Pior: {pior_custo:5d}")


            # Seleção
            if self.metodo_selecao == 'torneio':
                selecionados = Selecao.torneio_eficiente(self.populacao, fitness)  # <--- Corrigido
            elif self.metodo_selecao == 'roleta':
                selecionados = Selecao.roleta(self.populacao, fitness)
            else:
                raise ValueError(f"Método de seleção desconhecido: {self.metodo_selecao}")


            # Crossover
            filhos = []
            n_elites = max(1, int(self.taxa_elitismo * self.tamanho_populacao))
            
            # Garante que o número de crossovers seja par se o PMX retornar dois filhos
            num_crossovers = self.tamanho_populacao - n_elites
            if self.metodo_crossover == 'pmx' and not self.pmx_retorna_um_filho:
                num_crossovers = (num_crossovers // 2) * 2  # Arredonda para o número par mais próximo

            for _ in range(num_crossovers):
                pai1, pai2 = selecionados[np.random.randint(len(selecionados))], \
                            selecionados[np.random.randint(len(selecionados))]
                
                filho = self.aplicar_crossover(pai1, pai2)

                # Mutação
                if np.random.rand() < self.taxa_mutacao:
                    if self.metodo_mutacao == 'swap':
                        filho = Mutacao.swap(filho)
                    elif self.metodo_mutacao == 'inversao':
                        filho = Mutacao.inversao(filho)
                    else:
                        raise ValueError(f"Método de mutação desconhecido: {self.metodo_mutacao}")
                
                filhos.append(filho)

            # Elitismo
            if self.metodo_elitismo == 'top':
                elites = Elitismo.manter_melhores(self.populacao, fitness, n_elites)
            elif self.metodo_elitismo == 'hibrido':
                n_aleatorios = max(1, int(n_elites * 0.5))
                elites = Elitismo.manter_melhores_e_aleatorios(
                    self.populacao, fitness, n_elites, n_aleatorios)
            else:
                raise ValueError(f"Método de elitismo desconhecido: {self.metodo_elitismo}")

            # Completa a nova população com indivíduos aleatórios se necessário
            while len(filhos) < self.tamanho_populacao - n_elites:
                filhos.append(np.random.permutation(self.pqc.n).tolist())


            self.populacao = elites + filhos

            # Critério de parada
            if geracoes_sem_melhoria >= 50:
                print(f"Convergência na geração {geracao} (50 gerações sem melhoria)!")
                break

        return self.populacao[np.argmax(self._calcular_fitness())]  
