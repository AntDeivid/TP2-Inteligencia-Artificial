# Projeto de Algoritmos Genéticos para o Problema Quadrático de Alocação (PQA)

Este projeto implementa e executa experimentos utilizando **Algoritmos Genéticos** para resolver o **Problema Quadrático de Alocação (PQA)**. O objetivo é analisar o impacto de diferentes operadores genéticos, como **seleção, crossover, mutação e elitismo**, na busca por soluções eficientes.

## Estrutura do Projeto

- **algoritmo_genetico.py**: Implementação do Algoritmo Genético, incluindo os métodos de seleção, crossover, mutação e elitismo.
- **pqa.py**: Definição do Problema Quadrático de Alocação, incluindo a geração das matrizes de distância e fluxo e o cálculo da função de custo.
- **operadores.py**: Implementação dos operadores genéticos:
  - Seleção: Torneio e Roleta.
  - Crossover: OX e PMX.
  - Mutação: Swap e Inversão.
  - Elitismo: Top e Híbrido.
- **config.py**: Arquivo de configuração contendo parâmetros do algoritmo genético, como tamanho da população, taxa de mutação e elitismo.
- **experimentos/**: Diretório contendo os experimentos realizados:
  - **experimento_selecao.py**: Comparação dos métodos de seleção.
  - **experimento_crossover.py**: Comparação dos métodos de crossover.
  - **experimento_elitismo.py**: Comparação dos métodos de elitismo.\n  - **experimento_mutacao.py**: Comparação dos métodos de mutação.
  - **experimento_escala.py**: Teste de escalabilidade do algoritmo para diferentes tamanhos do problema.
- **resultados/**: Diretório onde os dados dos experimentos são salvos em arquivos CSV.

## Requisitos

- Python 3.x
- NumPy
- Pandas

### Instalação das Dependências

Execute o seguinte comando para instalar as bibliotecas necessárias:

```bash
pip install numpy pandas
```

## Como executar os experimentos
```bash
python experimento_elitismo.py
```
Os resultados serão salvos no diretório `resultados/`.

## Saída dos Experimentos
Cada execução do algoritmo genético imprime e salva os seguintes dados:
- Melhor fitness encontrado.
- Fitness médio da população.
- Fitness do pior indivíduo.
- Tempo de execução.

Os dados gerados são analisados e comparados para avaliar o impacto das diferentes variações dos operadores genéticos.

## Referências
Este projeto foi desenvolvido como parte do Trabalho 2 da disciplina de Inteligência Artificial - 2024.2, sob orientação do Prof. Samy Sá na Universidade Federal do Ceará (UFC) - Campus de Quixadá.