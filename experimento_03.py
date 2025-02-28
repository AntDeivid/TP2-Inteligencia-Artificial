# -*- coding: utf-8 -*-
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
    nome_arquivo = f"{pasta}/elitismo_{nome_experimento}_{timestamp}.csv"  # Prefixo adicionado
    
    fieldnames = ["execucao", "metodo_elitismo", "fitness"]  # Colunas simplificadas
    
    with open(nome_arquivo, mode="w", newline="", encoding="utf-8") as arquivo_csv:
        escritor = csv.DictWriter(arquivo_csv, fieldnames=fieldnames)
        escritor.writeheader()
        for item in dados:
            item["fitness"] = f"{item['fitness']:.6f}"  # Formato decimal
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
        fitness = 1 / (1 + pqc.calcular_custo(melhor_solucao))  # Fitness ao invés de custo
        
        resultados.append({
            "execucao": i + 1,
            "metodo_elitismo": metodo_elitismo,
            "fitness": fitness
        })
    
    return resultados

def main():
    pqc = PQA(n=10, seed=SEED)
    
    print("\n=== EXECUTANDO EXPERIMENTO - COMPARAÇÃO DE ELITISMO ===")
    
    resultados_top = executar_experimento_elitismo(pqc, 'top')
    resultados_hibrido = executar_experimento_elitismo(pqc, 'hibrido')
    
    salvar_dados_experimento("parte_3_elitismo", "top", resultados_top)
    salvar_dados_experimento("parte_3_elitismo", "hibrido", resultados_hibrido)
    
    fitness_top = [r["fitness"] for r in resultados_top]
    fitness_hibrido = [r["fitness"] for r in resultados_hibrido]
    
    print("\nRESULTADOS:")
    print(f"Elitismo TOP -> Média: {mean(fitness_top):.6f} | Desvio Padrão: {stdev(fitness_top):.6f}")
    print(f"Elitismo HÍBRIDO -> Média: {mean(fitness_hibrido):.6f} | Desvio Padrão: {stdev(fitness_hibrido):.6f}")

if __name__ == "__main__":
    main()