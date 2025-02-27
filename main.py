# -*- coding: utf-8 -*-
from pqa import PQA
from algoritmo_genetico import AlgoritmoGenetico
from config import *
import time

def executar_experimento(pqc, metodo_selecao, metodo_crossover, metodo_elitismo, metodo_mutacao, pmx_retorna_um_filho=False):
    inicio = time.time()

    ag = AlgoritmoGenetico(
        pqc=pqc,
        tamanho_populacao=TAMANHO_POPULACAO,
        max_geracoes=MAX_GERACOES,
        taxa_mutacao=TAXA_MUTACAO,
        taxa_elitismo=TAXA_ELITISMO,
        metodo_selecao=metodo_selecao,
        metodo_crossover=metodo_crossover,
        metodo_elitismo=metodo_elitismo,
        metodo_mutacao=metodo_mutacao,
        pmx_retorna_um_filho=pmx_retorna_um_filho
    )

    melhor_solucao = ag.executar()
    custo = pqc.calcular_custo(melhor_solucao)

    print(f"Config: {metodo_selecao.upper()}/{metodo_crossover.upper()}/{metodo_elitismo.upper()}/{metodo_mutacao.upper()}")
    print(f"Melhor custo: {custo} | Tempo: {time.time() - inicio:.2f}s\n")
    return custo

def main():
    # Configurar encoding para UTF-8
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    pqc = PQA(n=N, seed=SEED)
    #pqc = PQA(n=5) feito só para testa manualmente a logica :)
    #pqc.carregar_matrizes(
    #    distancias=[
    #        [0, 10, 15, 22, 25],
    #        [10, 0, 18, 12, 20],
    #        [15, 18, 0, 8, 14],
    #        [22, 12, 8, 0, 9],
    #        [25, 20, 14, 9, 0]
    #    ],
    #    fluxo=[
    #        [0, 3, 6, 2, 4],
    #        [3, 0, 1, 5, 7],
    #        [6, 1, 0, 3, 2],
    #        [2, 5, 3, 0, 4],
    #        [4, 7, 2, 4, 0]
    #    ]
    #)

    print("\n" + "="*50)
    print("INÍCIO DA EXECUÇÃO - TRABALHO 2 - ALGORITMOS GENÉTICOS")
    print("="*50 + "\n")

    # Parte 1: Comparação de Seleção
    print("\n=== PARTE 1: SELEÇÃO (TORNEIO vs ROLETA) ===")
    executar_experimento(pqc, 'torneio', 'ox', 'top', 'swap')
    executar_experimento(pqc, 'roleta', 'ox', 'top', 'swap')

    # #Parte 2: Comparação de Crossover
    # print("\n=== PARTE 2: CROSSOVER (OX vs PMX) ===")
    # executar_experimento(pqc, 'torneio', 'ox', 'top', 'swap')
    # executar_experimento(pqc, 'torneio', 'pmx', 'top', 'swap', pmx_retorna_um_filho=True)

    # # Parte 3: Comparação de Elitismo
    # print("\n=== PARTE 3: ELITISMO (TOP vs HIBRIDO) ===")
    # executar_experimento(pqc, 'torneio', 'ox', 'top', 'swap')
    # executar_experimento(pqc, 'torneio', 'ox', 'hibrido', 'swap')

    # # Parte 4: Comparação de Mutação
    # print("\n=== PARTE 4: MUTAÇÃO (SWAP vs INVERSÃO) ===")
    # executar_experimento(pqc, 'torneio', 'ox', 'top', 'swap')
    # executar_experimento(pqc, 'torneio', 'ox', 'top', 'inversao')

if __name__ == "__main__":
    main()