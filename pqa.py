import numpy as np
from typing import List, Tuple


class PQA:
    def __init__(self, n: int, seed: int = None):
        self.n = n
        if seed is not None:
            np.random.seed(seed)
        self.distancias, self.fluxo = self._gerar_entradas_aleatorias()
        self.fluxo_triangular = np.triu(self.fluxo, k=1)  # Pré-calcula o fluxo não repetido


    def _gerar_entradas_aleatorias(self) -> Tuple[np.ndarray, np.ndarray]:
        coordenadas = np.random.randint(0, 31, (self.n, 2))
        distancias = np.sqrt(np.sum((coordenadas[:, None, :] - coordenadas[None, :, :]) ** 2, axis=2))
        distancias = distancias = (distancias + distancias.T) // 2
        np.fill_diagonal(distancias, 0)


        fluxo = np.random.randint(0, 2*self.n + 1, (self.n, self.n))
        fluxo = (fluxo + fluxo.T) // 2
        np.fill_diagonal(fluxo, 0)


        return distancias, fluxo


    def calcular_custo(self, permutacao: List[int]) -> int:
        perm = np.array(permutacao)
        dist_perm = self.distancias[perm, :][:, perm]
        return np.sum(self.fluxo * dist_perm) // 2   # Otimização chave
    
    def carregar_matrizes(self, distancias, fluxo):
        self.distancias = np.array(distancias)
        self.fluxo = np.array(fluxo)
        self.fluxo_triangular = np.triu(self.fluxo, k=1)
