import datetime
import csv
import os
from statistics import mean, stdev
from pqa import PQA
from algoritmo_genetico import AlgoritmoGenetico
from config import *


def salvar_dados_experimento(parte, nome_experimento, dados):
    pasta = f"resultados/{parte}"
    os.makedirs(pasta, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"{pasta}/{nome_experimento}_{timestamp}.csv"

    fieldnames = ["execucao", "metodo_crossover", "melhor_custo"]  # Padronizado para 'custo'
    with open(nome_arquivo, mode="w", newline="") as arquivo_csv:
        escritor = csv.DictWriter(arquivo_csv, fieldnames=fieldnames)
        escritor.writeheader()
        for item in dados:
            escritor.writerow(item)

    print(f"Dados salvos em: {nome_arquivo}")


def executar_experimento_crossover(pqc, metodo_crossover):
    resultados = []
    for i in range(20):  # 20 execuções para consistência estatística
        ag = AlgoritmoGenetico(
            pqc=pqc,
            tamanho_populacao=TAMANHO_POPULACAO,
            max_geracoes=MAX_GERACOES,
            taxa_mutacao=TAXA_MUTACAO,
            taxa_elitismo=TAXA_ELITISMO,
            metodo_selecao='torneio',  # Seleção fixa (torneio)
            metodo_crossover=metodo_crossover,  # Variável (OX ou PMX)
            metodo_elitismo='top',  # Elitismo fixo (como no experimento_03)
            metodo_mutacao='swap',
            pmx_retorna_um_filho=True if metodo_crossover == 'pmx' else False
        )
        melhor_solucao = ag.executar()
        custo = pqc.calcular_custo(melhor_solucao)  # Padronizado para 'custo'
        resultados.append({
            "execucao": i + 1,
            "metodo_crossover": metodo_crossover,
            "melhor_custo": custo  # Campo renomeado
        })
    return resultados


def main():
    pqc = PQA(n=N, seed=SEED)  # Usar 'N' de config.py (como no experimento_03)
    print("\n=== EXECUTANDO EXPERIMENTO - COMPARAÇÃO DE MÉTODOS DE CROSSOVER ===")

    resultados_ox = executar_experimento_crossover(pqc, 'ox')
    resultados_pmx = executar_experimento_crossover(pqc, 'pmx')

    salvar_dados_experimento("parte_2_crossover", "crossover_ox", resultados_ox)
    salvar_dados_experimento("parte_2_crossover", "crossover_pmx", resultados_pmx)

    custos_ox = [r["melhor_custo"] for r in resultados_ox]
    custos_pmx = [r["melhor_custo"] for r in resultados_pmx]

    print("\nRESULTADOS DO EXPERIMENTO:")
    print(f"Média de custo OX: {mean(custos_ox):.1f} | Desvio padrão: {stdev(custos_ox):.1f}")
    print(f"Média de custo PMX: {mean(custos_pmx):.1f} | Desvio padrão: {stdev(custos_pmx):.1f}")

    melhor_metodo = "OX" if mean(custos_ox) < mean(custos_pmx) else "PMX"  # Menor custo é melhor
    print(f"\nMelhor método de crossover: {melhor_metodo}")

if __name__ == "__main__":
    main()
