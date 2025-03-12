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

    fieldnames = ["execucao", "metodo_mutacao", "melhor_custo"]  # Campo alterado para 'custo'
    with open(nome_arquivo, mode="w", newline="") as arquivo_csv:
        escritor = csv.DictWriter(arquivo_csv, fieldnames=fieldnames)
        escritor.writeheader()
        for item in dados:
            escritor.writerow(item)

    print(f"Dados salvos em: {nome_arquivo}")


def executar_experimento_mutacao(pqc, metodo_mutacao):
    resultados = []
    for i in range(20):  # 20 execuções para consistência estatística
        ag = AlgoritmoGenetico(
            pqc=pqc,
            tamanho_populacao=TAMANHO_POPULACAO,
            max_geracoes=MAX_GERACOES,
            taxa_mutacao=TAXA_MUTACAO,
            taxa_elitismo=TAXA_ELITISMO,
            metodo_selecao='torneio',  # Seleção fixa
            metodo_crossover='ox',      # Crossover fixo
            metodo_elitismo='top',      # Elitismo fixo
            metodo_mutacao=metodo_mutacao  # Variável (swap/inversao)
        )
        melhor_solucao = ag.executar()
        custo = pqc.calcular_custo(melhor_solucao)  # Usar custo diretamente
        resultados.append({
            "execucao": i + 1,
            "metodo_mutacao": metodo_mutacao,
            "melhor_custo": custo  # Campo renomeado
        })
    return resultados


def main():
    pqc = PQA(n=N, seed=SEED)  # Usar 'N' de config.py
    print("\n=== EXECUTANDO EXPERIMENTO - COMPARAÇÃO DE MÉTODOS DE MUTAÇÃO ===")

    resultados_swap = executar_experimento_mutacao(pqc, 'swap')
    resultados_inversao = executar_experimento_mutacao(pqc, 'inversao')

    salvar_dados_experimento("parte_4_mutacao", "mutacao_swap", resultados_swap)
    salvar_dados_experimento("parte_4_mutacao", "mutacao_inversao", resultados_inversao)

    custos_swap = [r["melhor_custo"] for r in resultados_swap]
    custos_inversao = [r["melhor_custo"] for r in resultados_inversao]

    print("\nRESULTADOS DO EXPERIMENTO:")
    print(f"Média de custo SWAP: {mean(custos_swap):.1f} | Desvio padrão: {stdev(custos_swap):.1f}")
    print(f"Média de custo INVERSÃO: {mean(custos_inversao):.1f} | Desvio padrão: {stdev(custos_inversao):.1f}")

    melhor_metodo = "SWAP" if mean(custos_swap) < mean(custos_inversao) else "INVERSÃO"  # Menor custo é melhor
    print(f"\nMelhor método de mutação: {melhor_metodo}")

if __name__ == "__main__":
    main()
