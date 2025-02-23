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

import numpy as np
from typing import List

class Crossover:
    @staticmethod
    def ox(pai1: List[int], pai2: List[int]) -> List[int]:
        """
        Order Crossover (OX) - Versão robusta para todos os casos.
        Mantém um segmento do pai1 e preenche o restante com a ordem do pai2.
        """
        size = len(pai1)
        if size <= 1:
            return pai1.copy()

        # 1. Seleciona pontos de corte distintos
        ponto1, ponto2 = sorted(np.random.choice(size, 2, replace=False))
        segmento = pai1[ponto1:ponto2]

        # 2. Inicializa o filho com o segmento do pai1
        filho = [-1] * size
        filho[ponto1:ponto2] = segmento

        # 3. Preenche posições vazias com genes do pai2 (ordem circular)
        ptr = ponto2  # Começa após o segmento
        for gene in pai2[ponto2:] + pai2[:ponto2]:
            if gene not in segmento:
                while filho[ptr % size] != -1:
                    ptr += 1
                filho[ptr % size] = gene
                ptr += 1

        # 4. Validação final
        if sorted(filho) != list(range(size)):
            raise ValueError(f"OX inválido: {filho}")
        return filho

    @staticmethod
    # não funciona direito ou trava o codigo ou da retorno de valor errado
    @staticmethod
    def pmx(pai1: List[int], pai2: List[int]) -> List[int]:
        size = len(pai1)
        if size <= 1:
            return pai1.copy()

        # 1. Seleciona pontos de corte distintos
        ponto1, ponto2 = sorted(np.random.choice(size, 2, replace=False))
        segmento_pai1 = pai1[ponto1:ponto2]

        # 2. Inicializa o filho com o segmento do pai1
        filho = [-1] * size
        filho[ponto1:ponto2] = segmento_pai1

        # 3. Mapeamento bidirecional entre os genes dos pais
        mapeamento = {}
        for i in range(ponto1, ponto2):
            gene_pai1 = pai1[i]
            gene_pai2 = pai2[i]
            mapeamento[gene_pai2] = gene_pai1  # Mapeia gene do pai2 para o pai1
            mapeamento[gene_pai1] = gene_pai2  # Mapeia gene do pai1 para o pai2

        # 4. Preenche todas as posições restantes
        for i in range(size):
            if filho[i] == -1:
                gene = pai2[i]
                # Resolve conflitos recursivamente até encontrar gene não mapeado
                while gene in mapeamento:
                    gene = mapeamento[gene]
                filho[i] = gene

        # 5. Validação final
        if sorted(filho) != list(range(size)):
            raise ValueError(f"PMX inválido: {filho}")
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
