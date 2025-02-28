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

    fieldnames = ["execucao", "metodo_crossover", "melhor_fitness"]
    with open(nome_arquivo, mode="w", newline="") as arquivo_csv:
        escritor = csv.DictWriter(arquivo_csv, fieldnames=fieldnames)
        escritor.writeheader()
        for item in dados:
            escritor.writerow(item)

    print(f"Dados salvos em: {nome_arquivo}")


def executar_experimento_crossover(pqc, metodo_crossover):
    """
    Executa um experimento comparando diferentes métodos de crossover.
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
            metodo_crossover=metodo_crossover,
            metodo_elitismo='top',
            metodo_mutacao='swap',
            pmx_retorna_um_filho=True if metodo_crossover == 'pmx' else False
        )

        melhor_solucao = ag.executar()
        fitness = 1 / (1 + pqc.calcular_custo(melhor_solucao))  # Fitness = 1 / (1 + custo)

        resultados.append({
            "execucao": i + 1,
            "metodo_crossover": metodo_crossover,
            "melhor_fitness": fitness
        })

    return resultados


def main():
    pqc = PQA(n=10, seed=SEED)

    print("\n=== EXECUTANDO EXPERIMENTO - COMPARAÇÃO DE CROSSOVER ===")

    # Executar experimentos com os métodos de crossover 'OX' e 'PMX'
    resultados_ox = executar_experimento_crossover(pqc, 'ox')
    resultados_pmx = executar_experimento_crossover(pqc, 'pmx')

    # Salvar resultados
    salvar_dados_experimento("parte_2_crossover", "crossover_ox", resultados_ox)
    salvar_dados_experimento("parte_2_crossover", "crossover_pmx", resultados_pmx)

    # Analisar resultados
    fitness_ox = [r["melhor_fitness"] for r in resultados_ox]
    fitness_pmx = [r["melhor_fitness"] for r in resultados_pmx]

    print("\nRESULTADOS DO EXPERIMENTO:")
    print(f"Média de fitness OX: {mean(fitness_ox):.6f} | Desvio padrão: {stdev(fitness_ox):.6f}")
    print(f"Média de fitness PMX: {mean(fitness_pmx):.6f} | Desvio padrão: {stdev(fitness_pmx):.6f}")

    melhor_metodo = "OX" if mean(fitness_ox) > mean(fitness_pmx) else "PMX"
    print(f"\nMétodo de crossover que apresentou melhor desempenho: {melhor_metodo}")

    print("\nExperimento concluído. Dados salvos com sucesso.")


if __name__ == "__main__":
    main()
