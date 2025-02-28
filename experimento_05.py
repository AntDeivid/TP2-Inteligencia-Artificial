import datetime
import csv
import os
import time
from statistics import mean, stdev
from pqa import PQA
from algoritmo_genetico import AlgoritmoGenetico
from config import *  # Certifique-se de que este arquivo contém as configurações padrão


def salvar_dados_experimento_tamanho(nome_experimento, dados):
    """
    Salva os dados do experimento de tamanho em um arquivo CSV.
    """
    pasta = "resultados/parte_5_tamanho"
    os.makedirs(pasta, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    nome_arquivo = f"{pasta}/{nome_experimento}_{timestamp}.csv"

    fieldnames = ["tamanho_entrada", "variacao", "tempo_execucao", "melhor_fitness"]  # Adicionado melhor_fitness
    with open(nome_arquivo, mode="w", newline="") as arquivo_csv:
        escritor = csv.DictWriter(arquivo_csv, fieldnames=fieldnames)
        escritor.writeheader()
        for item in dados:
            escritor.writerow(item)

    print(f"Dados salvos em: {nome_arquivo}")


def executar_variacao(pqc, variacao_nome, parametros):
    """
    Executa uma variação do algoritmo genético e retorna o tempo de execução.
    """
    inicio = time.time()
    ag = AlgoritmoGenetico(pqc=pqc, **parametros)
    melhor_solucao = ag.executar()
    fim = time.time()
    tempo_execucao = fim - inicio
    fitness = 1 / (1 + pqc.calcular_custo(melhor_solucao))
    return tempo_execucao, fitness


def main():
    tamanhos_entrada = range(10, 21, 1)  # Incrementa n de 10 a 20 (inclusive)
    resultados = []

    # Defina as 4 variações campeãs aqui.  Adaptar os parâmetros conforme necessário.
    variacoes = {
        "Variacao_1": {
            "tamanho_populacao": 100,  # Valor de exemplo
            "max_geracoes": 50,  # Valor de exemplo
            "taxa_mutacao": 0.1,  # Valor de exemplo
            "taxa_elitismo": 0.1,  # Valor de exemplo
            "metodo_selecao": 'torneio',  # Valor de exemplo
            "metodo_crossover": 'ox',  # Valor de exemplo
            "metodo_elitismo": 'top',  # Valor de exemplo
            "metodo_mutacao": 'swap'  # Valor de exemplo
        },
        "Variacao_2": {
            "tamanho_populacao": 150,  # Valor de exemplo
            "max_geracoes": 75,  # Valor de exemplo
            "taxa_mutacao": 0.2,  # Valor de exemplo
            "taxa_elitismo": 0.05,  # Valor de exemplo
            "metodo_selecao": 'roleta',  # Valor de exemplo
            "metodo_crossover": 'um_ponto',  # Valor de exemplo
            "metodo_elitismo": 'random',  # Valor de exemplo
            "metodo_mutacao": 'inversao'  # Valor de exemplo
        },
        "Variacao_3": {
            "tamanho_populacao": 80,  # Valor de exemplo
            "max_geracoes": 60,  # Valor de exemplo
            "taxa_mutacao": 0.05,  # Valor de exemplo
            "taxa_elitismo": 0.15,  # Valor de exemplo
            "metodo_selecao": 'torneio',  # Valor de exemplo
            "metodo_crossover": 'ciclo',  # Valor de exemplo
            "metodo_elitismo": 'top',  # Valor de exemplo
            "metodo_mutacao": 'swap'  # Valor de exemplo
        },
        "Variacao_4": {
            "tamanho_populacao": 120,  # Valor de exemplo
            "max_geracoes": 100,  # Valor de exemplo
            "taxa_mutacao": 0.15,  # Valor de exemplo
            "taxa_elitismo": 0.08,  # Valor de exemplo
            "metodo_selecao": 'roleta',  # Valor de exemplo
            "metodo_crossover": 'ox',  # Valor de exemplo
            "metodo_elitismo": 'random',  # Valor de exemplo
            "metodo_mutacao": 'inversao'  # Valor de exemplo
        }
    }

    print("\n=== EXECUTANDO EXPERIMENTO - TAMANHO MÁXIMO DE ENTRADA VIÁVEL ===")

    for n in tamanhos_entrada:
        print(f"\nExecutando para tamanho de entrada: {n}")
        pqc = PQA(n=n, seed=SEED)

        for variacao_nome, parametros in variacoes.items():
            print(f"  Executando variação: {variacao_nome}")
            try:
                tempo_execucao, fitness = executar_variacao(pqc, variacao_nome, parametros)  # Recebe fitness

                resultados.append({
                    "tamanho_entrada": n,
                    "variacao": variacao_nome,
                    "tempo_execucao": tempo_execucao,
                    "melhor_fitness": fitness  # Salva fitness
                })
                print(f"    Tempo de execução: {tempo_execucao:.4f} segundos, Melhor Fitness: {fitness:.6f}")
            except ValueError as e:
                print(f"    Erro durante a execução da variação {variacao_nome}: {e}")


    salvar_dados_experimento_tamanho("experimento_tamanho_entrada", resultados)

    print("\nExperimento concluído. Dados salvos com sucesso.")


if __name__ == "__main__":
    main()