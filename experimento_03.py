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

    fieldnames = ["execucao", "metodo_elitismo", "melhor_fitness"]
    with open(nome_arquivo, mode="w", newline="") as arquivo_csv:
        escritor = csv.DictWriter(arquivo_csv, fieldnames=fieldnames)
        escritor.writeheader()
        for item in dados:
            escritor.writerow(item)

    print(f"Dados salvos em: {nome_arquivo}")


def executar_experimento_elitismo(pqc, metodo_elitismo):
    """
    Executa um experimento comparando diferentes métodos de elitismo.
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
            metodo_elitismo=metodo_elitismo,
            metodo_mutacao='swap'
        )

        melhor_solucao = ag.executar()
        fitness = 1 / (1 + pqc.calcular_custo(melhor_solucao))  # Fitness = 1 / (1 + custo)

        resultados.append({
            "execucao": i + 1,
            "metodo_elitismo": metodo_elitismo,
            "melhor_fitness": fitness
        })

    return resultados


def main():
    pqc = PQA(n=10, seed=SEED)

    print("\n=== EXECUTANDO EXPERIMENTO - COMPARAÇÃO DE ELITISMO ===")

    # Executar experimentos com os métodos de elitismo 'top' e 'híbrido'
    resultados_top = executar_experimento_elitismo(pqc, 'top')
    resultados_hibrido = executar_experimento_elitismo(pqc, 'hibrido')

    # Salvar resultados
    salvar_dados_experimento("parte_3_elitismo", "elitismo_top", resultados_top)
    salvar_dados_experimento("parte_3_elitismo", "elitismo_hibrido", resultados_hibrido)

    # Analisar resultados
    fitness_top = [r["melhor_fitness"] for r in resultados_top]
    fitness_hibrido = [r["melhor_fitness"] for r in resultados_hibrido]

    print("\nRESULTADOS DO EXPERIMENTO:")
    print(f"Média de fitness Top: {mean(fitness_top):.6f} | Desvio padrão: {stdev(fitness_top):.6f}")
    print(f"Média de fitness Híbrido: {mean(fitness_hibrido):.6f} | Desvio padrão: {stdev(fitness_hibrido):.6f}")

    melhor_metodo = "Top" if mean(fitness_top) > mean(fitness_hibrido) else "Híbrido"
    print(f"\nMétodo de elitismo que apresentou melhor desempenho: {melhor_metodo}")

    print("\nExperimento concluído. Dados salvos com sucesso.")


if __name__ == "__main__":
    main()
