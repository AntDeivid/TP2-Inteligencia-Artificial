import datetime
import csv
import os
import time
from statistics import mean, stdev
from pqa import PQA
from algoritmo_genetico import AlgoritmoGenetico
from config import *


def salvar_dados_experimento(parte, nome_experimento, dados):
    """
    Salva os dados do experimento em um arquivo CSV dentro da pasta correspondente à parte.
    """
    pasta = f"resultados/{parte}"
    os.makedirs(pasta, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"{pasta}/{nome_experimento}_{timestamp}.csv"

    fieldnames = ["execucao", "metodo_mutacao", "melhor_fitness"]
    with open(nome_arquivo, mode="w", newline="") as arquivo_csv:
        escritor = csv.DictWriter(arquivo_csv, fieldnames=fieldnames)
        escritor.writeheader()
        for item in dados:
            escritor.writerow(item)

    print(f"Dados salvos em: {nome_arquivo}")


def executar_experimento_mutacao(pqc, metodo_mutacao):
    """
    Executa um experimento comparando diferentes métodos de mutação.
    """
    resultados = []

    for i in range(20):  # Executar 20 vezes
        ag = AlgoritmoGenetico(
            pqc=pqc,
            tamanho_populacao=TAMANHO_POPULACAO,
            max_geracoes=MAX_GERACOES,
            taxa_mutacao=TAXA_MUTACAO,
            taxa_elitismo=TAXA_ELITISMO,
            metodo_selecao='torneio',
            metodo_crossover='ox',
            metodo_elitismo='top',
            metodo_mutacao=metodo_mutacao
        )

        melhor_solucao = ag.executar()
        fitness = 1 / (1 + pqc.calcular_custo(melhor_solucao))  # Fitness = 1 / (1 + custo)

        resultados.append({
            "execucao": i + 1,
            "metodo_mutacao": metodo_mutacao,
            "melhor_fitness": fitness
        })

    return resultados


def main():
    pqc = PQA(n=10, seed=SEED)

    print("\n=== EXECUTANDO EXPERIMENTO - COMPARAÇÃO DE MUTAÇÃO ===")

    # Executar experimentos com os métodos de mutação 'Swap' e 'Inversão'
    resultados_swap = executar_experimento_mutacao(pqc, 'swap')
    resultados_inversao = executar_experimento_mutacao(pqc, 'inversao')

    # Salvar resultados
    salvar_dados_experimento("parte_4_mutacao", "mutacao_swap", resultados_swap)
    salvar_dados_experimento("parte_4_mutacao", "mutacao_inversao", resultados_inversao)

    # Analisar resultados
    fitness_swap = [r["melhor_fitness"] for r in resultados_swap]
    fitness_inversao = [r["melhor_fitness"] for r in resultados_inversao]

    print("\nRESULTADOS DO EXPERIMENTO:")
    print(f"Média de fitness SWAP: {mean(fitness_swap):.6f} | Desvio padrão: {stdev(fitness_swap):.6f}")
    print(f"Média de fitness INVERSÃO: {mean(fitness_inversao):.6f} | Desvio padrão: {stdev(fitness_inversao):.6f}")

    melhor_metodo = "SWAP" if mean(fitness_swap) > mean(fitness_inversao) else "INVERSÃO"
    print(f"\nMétodo de mutação que apresentou melhor desempenho: {melhor_metodo}")

    print("\nExperimento concluído. Dados salvos com sucesso.")


if __name__ == "__main__":
    main()
