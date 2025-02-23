import numpy as np
from typing import List

class Selecao:
    @staticmethod

    def roleta(populacao: List[List[int]], fitness: List[float]) -> List[List[int]]:
        total_fitness = sum(fitness)
        if total_fitness == 0:  # Evitar divisão por zero
            return populacao.copy()
        probabilidades = [f / total_fitness for f in fitness]
        indices = np.random.choice(len(populacao), size=len(populacao), p=probabilidades)
        return [populacao[i] for i in indices]

    ##//////////////////////////

    def torneio_eficiente(populacao: List[List[int]], fitness: List[float]) -> List[List[int]]:
        indices = np.random.randint(0, len(populacao), (len(populacao), 3))  # 3 participantes por torneio
        fitness_participantes = np.array(fitness)[indices]
        vencedores = indices[np.arange(len(populacao)), np.argmax(fitness_participantes, axis=1)]
        return [populacao[i] for i in vencedores]

class Crossover:
    @staticmethod


    def pmx(pai1: List[int], pai2: List[int]) -> List[int]:
        size = len(pai1)
        ponto1, ponto2 = sorted(np.random.choice(size, 2, replace=False))
        filho = [-1] * size
        filho[ponto1:ponto2] = pai1[ponto1:ponto2]
        mapping = {}

        # Mapeamento entre os pais
        for i in range(ponto1, ponto2):
            gene_pai2 = pai2[i]
            if gene_pai2 not in filho:
                current = gene_pai2
                while current in pai1[ponto1:ponto2]:
                    current = pai2[pai1.index(current)]
                mapping[gene_pai2] = current

        # Preenche o filho
        for i in range(size):
            if filho[i] == -1:
                gene = pai2[i]
                while gene in mapping:
                    gene = mapping[gene]
                filho[i] = gene
        return filho

    ##////////////////////////
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
    @staticmethod
    def inversao(individuo: List[int]) -> List[int]:
        i, j = sorted(np.random.choice(len(individuo), 2, replace=False))
        return individuo[:i] + list(reversed(individuo[i:j])) + individuo[j:]


# operadores.py (classe Elitismo corrigida)
class Elitismo:
    @staticmethod
    def manter_melhores(populacao: List[List[int]], fitness: List[float], n_elites: int) -> List[List[int]]:
        indices_ordenados = np.argsort(fitness)[::-1]  # Ordena em ordem decrescente de fitness
        return [populacao[i] for i in indices_ordenados[:n_elites]]

    @staticmethod
    def manter_melhores_e_aleatorios(populacao: List[List[int]], fitness: List[float], 
                                    n_elites: int, n_aleatorios: int) -> List[List[int]]:
        indices_ordenados = np.argsort(fitness)[::-1]
        elites = [populacao[i] for i in indices_ordenados[:n_elites]]
        aleatorios = [populacao[i] for i in np.random.choice(len(populacao), n_aleatorios, replace=False)]
        return elites + aleatorios
