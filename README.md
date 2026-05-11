# Laboratorio 9 - RAG com HNSW, HyDE e Cross-Encoder

Este repositorio implementa um pipeline de Retrieval-Augmented Generation para busca em fragmentos de manuais medicos privados.

## O que foi implementado

- Base simulada com 24 trechos tecnicos de saude.
- Transformacao HyDE da pergunta coloquial para um documento hipotetico.
- Vetorizacao dos documentos e da resposta hipotetica.
- Indice vetorial FAISS com HNSW.
- Recuperacao Top-10 por similaridade de cosseno.
- Re-ranking dos 10 documentos com Cross-Encoder.
- Impressao dos Top-3 finais.

## Como rodar

Instale as dependencias:

```bash
pip install -r requirements.txt
```

Execute com a pergunta padrao:

```bash
python rag_hyde_hnsw.py
```

Ou passe uma pergunta propria:

```bash
python rag_hyde_hnsw.py --query "dor de cabeca latejante e luz incomodando"
```

Para tentar usar OpenAI na etapa HyDE, defina `OPENAI_API_KEY` e rode:

```bash
python rag_hyde_hnsw.py --use-openai-hyde
```

## HNSW: M, ef_construction e memoria

O HNSW cria um grafo aproximado de vizinhos para acelerar a busca. Em vez de comparar a query com todos os vetores como no KNN exato, o algoritmo navega por camadas do grafo ate chegar a candidatos bons.

O parametro `M` controla quantas conexoes cada no tende a manter. Quanto maior o `M`, mais caminhos o grafo tem, o que geralmente melhora recall e estabilidade da busca. O custo e direto: mais arestas guardadas por vetor, logo mais memoria RAM consumida.

O `ef_construction` controla o tamanho da lista de candidatos avaliada durante a construcao do grafo. Valores maiores produzem um indice melhor conectado e com recuperacao mais precisa, mas a construcao fica mais lenta e pode usar mais memoria temporaria durante a indexacao.

Comparando com KNN exato, o HNSW normalmente usa memoria extra porque guarda nao so os vetores, mas tambem a estrutura do grafo. A vantagem e que, depois de construido, ele evita varrer a base inteira a cada consulta. Em bases pequenas, como os 24 documentos deste laboratorio, a diferenca pratica e pequena; em bases grandes, o ganho de latencia costuma compensar a memoria adicional.

## Nota

Partes deste laboratorio foram apoiadas por IA em questoes de logica, dificuldades pontuais na correcao de bugs e testes em escala para validar o funcionamento do codigo, com revisao e validacao final por Andreas Carvalho.
