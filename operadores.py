import numpy as np
from typing import List

class Selecao:
    @staticmethod
    def torneio_eficiente(populacao: List[List[int]], fitness: List[float]) -> List[List[int]]:
        indices = np.random.randint(0, len(populacao), (len(populacao), 3))  # 3 participantes por torneio
        fitness_participantes = np.array(fitness)[indices]
        vencedores = indices[np.arange(len(populacao)), np.argmax(fitness_participantes, axis=1)]
        return [populacao[i] for i in vencedores]

class Crossover:
    @staticmethod
    def ox(pai1: List[int], pai2: List[int]) -> List[int]:
        size = len(pai1)
        ponto1, ponto2 = sorted(np.random.choice(size, 2, replace=False))
        segmento = pai1[ponto1:ponto2]
        filho = [-1] * size
        filho[ponto1:ponto2] = segmento
        ptr = ponto2 % size
        for gene in pai2[ponto2:] + pai2[:ponto2]:  
            if gene not in segmento:
                filho[ptr] = gene
                ptr = (ptr + 1) % size
                if ptr == ponto1:
                    break
        return filho

class Mutacao:
    @staticmethod
    def swap(individuo: List[int]) -> List[int]:
        novo_ind = individuo.copy()
        i, j = np.random.choice(len(individuo), 2, replace=False)
        novo_ind[i], novo_ind[j] = novo_ind[j], novo_ind[i]
        return novo_ind

class Elitismo:
    @staticmethod
    def manter_melhores(populacao: List[List[int]], fitness: List[float], n_elites: int) -> List[List[int]]:
        return [populacao[i] for i in np.argsort(fitness)[-n_elites:]]