# -*- coding: utf-8 -*-
# experimento_03.py
import datetime
import csv
import os
import time
from statistics import mean, stdev
from pqa import PQA
from algoritmo_genetico import AlgoritmoGenetico
from config import *

def salvar_dados_experimento(parte, nome_experimento, dados):
    pasta = f"resultados/{parte}"
    os.makedirs(pasta, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"{pasta}/{nome_experimento}_{timestamp}.csv"

    fieldnames = ["execucao", "metodo_elitismo", "melhor_custo"]
    with open(nome_arquivo, mode="w", newline="", encoding='utf-8') as arquivo_csv:
        escritor = csv.DictWriter(arquivo_csv, fieldnames=fieldnames)
        escritor.writeheader()
        for item in dados:
            escritor.writerow(item)

    print(f"Dados salvos em: {nome_arquivo}")

def executar_experimento_elitismo(pqc, metodo_elitismo):
    resultados = []
    for i in range(20):
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
        custo = pqc.calcular_custo(melhor_solucao)
        resultados.append({"execucao": i + 1, "metodo_elitismo": metodo_elitismo, "melhor_custo": custo})
    return resultados

def main():
    pqc = PQA(n=N, seed=SEED)
    print("\n=== EXECUTANDO EXPERIMENTO - COMPARAÇÃO DE ELITISMO ===")
    resultados_top = executar_experimento_elitismo(pqc, 'top')
    resultados_hibrido = executar_experimento_elitismo(pqc, 'hibrido')
    salvar_dados_experimento("parte_3_elitismo", "elitismo_top", resultados_top)
    salvar_dados_experimento("parte_3_elitismo", "elitismo_hibrido", resultados_hibrido)

    custos_top = [r["melhor_custo"] for r in resultados_top]
    custos_hibrido = [r["melhor_custo"] for r in resultados_hibrido]

    print("\nRESULTADOS DO EXPERIMENTO:")
    print(f"Média de custo Top: {mean(custos_top):.1f} | Desvio padrão: {stdev(custos_top):.1f}")
    print(f"Média de custo Híbrido: {mean(custos_hibrido):.1f} | Desvio padrão: {stdev(custos_hibrido):.1f}")
    melhor_metodo = "Top" if mean(custos_top) < mean(custos_hibrido) else "Híbrido"
    print(f"\nMelhor método: {melhor_metodo}")

if __name__ == "__main__":
    main()
