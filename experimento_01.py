# -*- coding: utf-8 -*-
import datetime
import csv
import os
import time
import statistics
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
    
    fieldnames = ["execucao", "metodo_selecao", "melhor_individuo", "fitness"]
    with open(nome_arquivo, mode="w", newline="") as arquivo_csv:
        escritor = csv.DictWriter(arquivo_csv, fieldnames=fieldnames)
        escritor.writeheader()
        for item in dados:
            escritor.writerow(item)
    
    print(f"Dados salvos em: {nome_arquivo}")

def executar_experimento(pqc, metodo_selecao):
    """
    Executa um experimento comparando diferentes métodos de seleção.
    """
    resultados = []
    
    for i in range(20):  # Executar 20 vezes
        ag = AlgoritmoGenetico(
            pqc=pqc,
            tamanho_populacao=TAMANHO_POPULACAO,
            max_geracoes=MAX_GERACOES,
            taxa_mutacao=TAXA_MUTACAO,
            taxa_elitismo=TAXA_ELITISMO,
            metodo_selecao=metodo_selecao,
            metodo_crossover='ox',
            metodo_elitismo='top',
            metodo_mutacao='swap'
        )
        
        melhor_solucao = ag.executar()
        fitness = pqc.calcular_custo(melhor_solucao)
        
        resultados.append({
            "execucao": i + 1,
            "metodo_selecao": metodo_selecao,
            "melhor_individuo": melhor_solucao,
            "fitness": fitness
        })
    
    return resultados

def main():
    pqc = PQA(n=N, seed=SEED)  # Alterado para usar N do config.py
    
    print("\n=== EXECUTANDO EXPERIMENTO - COMPARAÇÃO DE SELEÇÃO ===")
    
    # Executar experimentos com os métodos de seleção 'torneio' e 'roleta'
    resultados_torneio = executar_experimento(pqc, 'torneio')
    resultados_roleta = executar_experimento(pqc, 'roleta')
    
    # Salvar resultados
    salvar_dados_experimento("parte_1_selecao", "selecao_torneio", resultados_torneio)
    salvar_dados_experimento("parte_1_selecao", "selecao_roleta", resultados_roleta)

    # Análise estatística (trecho adicionado)
    custos_torneio = [r["fitness"] for r in resultados_torneio]
    custos_roleta = [r["fitness"] for r in resultados_roleta]

    print("\nRESULTADOS DO EXPERIMENTO:")
    print(f"Média de custo Torneio: {statistics.mean(custos_torneio):.1f} | Desvio padrão: {statistics.stdev(custos_torneio):.1f}")
    print(f"Média de custo Roleta: {statistics.mean(custos_roleta):.1f} | Desvio padrão: {statistics.stdev(custos_roleta):.1f}")
    
    melhor_metodo = "Torneio" if statistics.mean(custos_torneio) < statistics.mean(custos_roleta) else "Roleta"
    print(f"\nMelhor método de seleção: {melhor_metodo}")

    print("\nExperimento concluído. Dados salvos com sucesso.")
    
if __name__ == "__main__":
    main()
