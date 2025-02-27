import numpy as np
from typing import List

class Selecao:
    @staticmethod
    def roleta(populacao: List[List[int]], fitness: List[float]) -> List[List[int]]:
        total_fitness = sum(fitness)

        # Se todos os fitness forem iguais ou zero, escolhe aleatoriamente
        if total_fitness == 0 or len(set(fitness)) == 1:
            return [populacao[i] for i in np.random.choice(len(populacao), size=len(populacao), replace=True)]

        # Normaliza probabilidades
        probabilidades = [f / total_fitness for f in fitness]
        indices = np.random.choice(len(populacao), size=len(populacao), p=probabilidades)
        return [populacao[i] for i in indices]


    ##//////////////////////////

    @staticmethod
    def torneio_eficiente(populacao: List[List[int]], fitness: List[float]) -> List[List[int]]:
        indices = np.random.randint(0, len(populacao), (len(populacao), 3))  # 3 participantes por torneio
        fitness_participantes = np.array(fitness)[indices]
        vencedores = indices[np.arange(len(populacao)), np.argmax(fitness_participantes, axis=1)]
        return [populacao[i] for i in vencedores]

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
    def pmx(pai1: List[int], pai2: List[int]) -> List[List[int]]:
        """
        PMX (Partially Mapped Crossover).  Retorna dois filhos.
        """
        size = len(pai1)
        assert len(pai1) == len(pai2), "Pais devem ter o mesmo tamanho"

        # 1. Escolher dois pontos de corte aleatórios
        ponto1, ponto2 = sorted(np.random.choice(size, 2, replace=False))

        # Função auxiliar para criar um filho a partir dos pais
        def criar_filho(p1, p2):
            filho = p1[:]  # Inicializa com uma cópia do primeiro pai

            # Copia o segmento entre os pontos de corte do primeiro pai para o filho
            filho[ponto1:ponto2] = p1[ponto1:ponto2]

            # Mapeamento
            mapeamento = {}
            for i in range(ponto1, ponto2):
                mapeamento[p1[i]] = p2[i]

            # Preencher as posições fora dos pontos de corte usando o pai2 e o mapeamento
            for i in range(size):
                if i < ponto1 or i >= ponto2:
                    gene = p2[i]
                    while gene in filho[ponto1:ponto2]:
                        gene = mapeamento[gene]  # Segue o mapeamento
                    filho[i] = gene
            return filho

        # Criar os dois filhos
        filho1 = criar_filho(pai1, pai2)
        filho2 = criar_filho(pai2, pai1)

        return filho1, filho2


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