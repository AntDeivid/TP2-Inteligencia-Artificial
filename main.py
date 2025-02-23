# -*- coding: utf-8 -*-
from pqa import PQA
from algoritmo_genetico import AlgoritmoGenetico
from config import *


def main():
    # Configurar encoding para UTF-8
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
   
    # Inicialização
    # No main.py:
    #pqc = PQA(n=3)
    #pqc.carregar_matrizes(
     #   distancias=[[0, 10, 15], [10, 0, 12], [15, 12, 0]],
      #  fluxo=[[0, 3, 6], [3, 0, 1], [6, 1, 0]]
#)


    pqc = PQA(n=N, seed=SEED)
    ag = AlgoritmoGenetico(
        pqc=pqc,
        tamanho_populacao=TAMANHO_POPULACAO,
        max_geracoes=MAX_GERACOES,
        taxa_mutacao=TAXA_MUTACAO,
        taxa_elitismo=TAXA_ELITISMO
    )


    # Execução
    melhor_solucao = ag.executar()
   
    # Resultado Final
    print("\n=== Resultado Final ===")
    print(f"Melhor solução encontrada: {melhor_solucao}")
    print(f"Custo: {pqc.calcular_custo(melhor_solucao)}")
   
    # Matrizes (opcional, conforme exigência do professor)
    print("\nMatriz de Distâncias:")
    print(pqc.distancias)
    print("\nMatriz de Fluxo:")
    print(pqc.fluxo)


if __name__ == "__main__":
    main()
