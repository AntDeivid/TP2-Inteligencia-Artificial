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
    
    Parâmetros:
      parte: Nome da parte do experimento (ex: "parte_1_selecao").
      nome_experimento: Nome identificador para os dados (ex: "selecao").
      dados: Lista de dicionários com os resultados.
    """
    pasta = f"resultados/{parte}"
    os.makedirs(pasta, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"{pasta}/{nome_experimento}_{timestamp}.csv"
    
    fieldnames = ["config", "custo", "tempo", "media", "desvio_padrao", "tempo_total"]
    with open(nome_arquivo, mode="w", newline="") as arquivo_csv:
        escritor = csv.DictWriter(arquivo_csv, fieldnames=fieldnames)
        escritor.writeheader()
        for item in dados:
            escritor.writerow(item)
    
    print(f"Dados salvos em: {nome_arquivo}")
    
def testar_parametros():
    """Executa testes variando parâmetros para encontrar configurações eficientes."""
    tamanhos_populacao = [50, 100, 200, 500]
    max_geracoes = [100, 200, 500]
    taxas_mutacao = [0.05, 0.1, 0.2]
    taxas_elitismo = [0.05, 0.1, 0.2]

    melhor_config = None
    melhor_custo = float('inf')

    for pop in tamanhos_populacao:
        for gen in max_geracoes:
            for mut in taxas_mutacao:
                for elit in taxas_elitismo:
                    pqc = PQA(n=10, seed=42)  # Tamanho fixo para experimentação
                    ag = AlgoritmoGenetico(
                        pqc=pqc,
                        tamanho_populacao=pop,
                        max_geracoes=gen,
                        taxa_mutacao=mut,
                        taxa_elitismo=elit,
                        metodo_selecao='torneio',
                        metodo_crossover='ox',
                        metodo_elitismo='top',
                        metodo_mutacao='swap'
                    )

                    inicio = time.time()
                    solucao = ag.executar()
                    custo = pqc.calcular_custo(solucao)
                    tempo_execucao = time.time() - inicio

                    print(
                        f"População: {pop}, Gerações: {gen}, Mutação: {mut}, Elitismo: {elit} | Custo: {custo} | Tempo: {tempo_execucao:.2f}s")

                    if custo < melhor_custo:
                        melhor_custo = custo
                        melhor_config = (pop, gen, mut, elit)

    print("\nMelhor configuração encontrada:")
    print(
        f"População: {melhor_config[0]}, Gerações: {melhor_config[1]}, Mutação: {melhor_config[2]}, Elitismo: {melhor_config[3]}")
    return melhor_config


def testar_tamanho_maximo():
    """Aumenta o tamanho da entrada PQA até que o tempo de execução fique impraticável."""
    tamanhos_n = [10, 20, 50, 100, 200, 500]  # Ajuste conforme necessário
    tempo_limite = 60  # Limite de tempo aceitável por execução (em segundos)

    for n in tamanhos_n:
        pqc = PQA(n=n, seed=42)
        ag = AlgoritmoGenetico(
            pqc=pqc,
            tamanho_populacao=100,  # Parâmetro fixo baseado em testes anteriores
            max_geracoes=100,
            taxa_mutacao=0.05,
            taxa_elitismo=0.1,
            metodo_selecao='torneio',
            metodo_crossover='ox',
            metodo_elitismo='top',
            metodo_mutacao='swap'
        )

        inicio = time.time()
        ag.executar()
        tempo_execucao = time.time() - inicio

        print(f"Tamanho n={n} | Tempo: {tempo_execucao:.2f}s")

        if tempo_execucao > tempo_limite:
            print(f"\nTamanho máximo viável identificado: {n // 2}")
            break

def executar_experimento(pqc, metodo_selecao, metodo_crossover, metodo_elitismo, metodo_mutacao, pmx_retorna_um_filho=False):
    tempos_execucao = []
    custos = []
    
    for _ in range(5):
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
        tempo_execucao = time.time() - inicio
        
        custos.append(custo)
        tempos_execucao.append(tempo_execucao)
    
    media = statistics.mean(custos)
    desvio_padrao = statistics.stdev(custos) if len(custos) > 1 else 0
    tempo_total = sum(tempos_execucao)
    
    config = f"{metodo_selecao}/{metodo_crossover}/{metodo_elitismo}/{metodo_mutacao}"
    print(f"Config: {config}")
    print(f"Média: {media} | Desvio Padrão: {desvio_padrao} | Tempo Total: {tempo_total:.2f}s\n")
    
    return {"config": config, "custo": media, "tempo": f"{statistics.mean(tempos_execucao):.2f}s", "media": media, "desvio_padrao": desvio_padrao, "tempo_total": f"{tempo_total:.2f}s"}

def main():
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    pqc = PQA(n=N, seed=SEED)

    print("\n" + "="*50)
    print("INÍCIO DA EXECUÇÃO - TRABALHO 2 - ALGORITMOS GENÉTICOS")
    print("="*50 + "\n")

    # Parte 0: Escolha de Parâmetros
    print("\n=== PARTE 0: ESCOLHA DE PARÂMETROS ===")
    melhor_config = testar_parametros()
    print("\n")
    print(f"Melhor configuração encontrada:")
    print(f"População: {melhor_config[0]}, Gerações: {melhor_config[1]}, Mutação: {melhor_config[2]}, Elitismo: {melhor_config[3]}")
    print("\n")
    
    # --- PARTE 1: Comparação de Seleção ---
    # print("\n=== PARTE 1: SELEÇÃO (TORNEIO vs ROLETA) ===")
    # resultados_parte1 = []
    # resultados_parte1.append(executar_experimento(pqc, 'torneio', 'ox', 'top', 'swap'))
    # resultados_parte1.append(executar_experimento(pqc, 'roleta', 'ox', 'top', 'swap'))
    # salvar_dados_experimento("parte_1_selecao", "selecao", resultados_parte1)

    # # --- PARTE 2: Comparação de Crossover ---
    # print("\n=== PARTE 2: CROSSOVER (OX vs PMX) ===")
    # resultados_parte2 = []
    # resultados_parte2.append(executar_experimento(pqc, 'torneio', 'ox', 'top', 'swap'))
    # resultados_parte2.append(executar_experimento(pqc, 'torneio', 'pmx', 'top', 'swap', pmx_retorna_um_filho=True))
    # salvar_dados_experimento("parte_2_crossover", "crossover", resultados_parte2)

    # # --- PARTE 3: Comparação de Elitismo ---
    # print("\n=== PARTE 3: ELITISMO (TOP vs HÍBRIDO) ===")
    # resultados_parte3 = []
    # resultados_parte3.append(executar_experimento(pqc, 'torneio', 'ox', 'top', 'swap'))
    # resultados_parte3.append(executar_experimento(pqc, 'torneio', 'ox', 'hibrido', 'swap'))
    # salvar_dados_experimento("parte_3_elitismo", "elitismo", resultados_parte3)

    # #--- PARTE 4: Comparação de Mutação ---
    # print("\n=== PARTE 4: MUTAÇÃO (SWAP vs INVERSÃO) ===")
    # resultados_parte4 = []
    # resultados_parte4.append(executar_experimento(pqc, 'torneio', 'ox', 'top', 'swap'))
    # resultados_parte4.append(executar_experimento(pqc, 'torneio', 'ox', 'top', 'inversao'))
    # salvar_dados_experimento("parte_4_mutacao", "mutacao", resultados_parte4)
    
    # print("\n=== PARTE 5: TAMANHO MÁXIMO DE ENTRADA VIÁVEL ===")
    # testar_tamanho_maximo()

if __name__ == "__main__":
    main()
