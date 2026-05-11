from __future__ import annotations

import argparse
import os
import textwrap
from dataclasses import dataclass

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DOCUMENTOS = [
    "Cefaleia migranosa: dor de cabeca unilateral e pulsatil, frequentemente relatada como latejante, agravada por luz intensa, associada a fotofobia, fonofobia, nauseas e possivel aura visual.",
    "Cefaleia tensional: dor bilateral em aperto, intensidade leve a moderada, sem piora importante ao esforco e sem sinais neurologicos focais.",
    "Rinossinusite aguda: obstrucao nasal, rinorreia purulenta, dor facial em pressao e reducao do olfato por periodo inferior a quatro semanas.",
    "Broncoespasmo asmatico: dispneia episodica, sibilancia expiratoria, tosse seca noturna e variacao do pico de fluxo expiratorio.",
    "Pneumonia adquirida na comunidade: febre, tosse produtiva, taquipneia, crepitacoes localizadas e infiltrado novo em exame de imagem.",
    "Refluxo gastroesofagico: pirose retroesternal, regurgitacao acida, piora pos-prandial e sintomas ao decubito dorsal.",
    "Dispepsia funcional: epigastralgia recorrente, plenitude pos-prandial e saciedade precoce sem lesao organica evidente.",
    "Colelitiase sintomatica: colica biliar em hipocondrio direito, irradiacao escapular, nausea e relacao com refeicoes gordurosas.",
    "Infeccao urinaria baixa: disuria, polaciuria, urgencia miccional, dor suprapubica e ausencia de sinais sistemicos importantes.",
    "Pielonefrite aguda: febre, calafrios, dor lombar, nauseas e sinal de Giordano positivo associado a bacteriuria.",
    "Dermatite atopica: prurido cronico, xerose cutanea, lesoes eczematosas em areas flexurais e historia pessoal de atopia.",
    "Urticaria aguda: placas eritematoedematosas migratorias, pruriginosas, com duracao inferior a 24 horas por lesao individual.",
    "Hipoglicemia: sudorese fria, tremores, palpitacoes, confusao mental e melhora apos ingestao de carboidrato.",
    "Diabetes mellitus descompensado: poliuria, polidipsia, perda ponderal, fadiga e hiperglicemia persistente.",
    "Hipertensao arterial sistemica: elevacao sustentada da pressao arterial, frequentemente assintomatica, com risco cardiovascular aumentado.",
    "Acidente vascular cerebral isquemico: deficit neurologico focal subito, assimetria facial, disartria, hemiparesia e janela terapeutica limitada.",
    "Vertigem posicional paroxistica benigna: vertigem rotatoria breve, desencadeada por mudanca de posicao da cabeca, sem perda auditiva.",
    "Otite media aguda: otalgia, febre, membrana timpanica hiperemiada e abaulada, comum apos infeccao de vias aereas superiores.",
    "Conjuntivite viral: hiperemia ocular, lacrimejamento, sensacao de areia, secrecao serosa e alta transmissibilidade.",
    "Anemia ferropriva: fadiga, palidez cutaneomucosa, unhas frageis, ferritina baixa e microcitose em hemograma.",
    "Trombose venosa profunda: edema unilateral de membro inferior, dor em panturrilha, empastamento e fatores de risco trombotico.",
    "Insuficiencia cardiaca congestiva: dispneia aos esforcos, ortopneia, edema periferico, turgencia jugular e estertores bibasais.",
    "Crise de ansiedade: taquicardia, tremores, sensacao de falta de ar, medo intenso e parestesias periorais sem causa cardiopulmonar evidente.",
    "Apendicite aguda: dor migratoria para fossa iliaca direita, anorexia, nauseas, febre baixa e irritacao peritoneal localizada.",
]


@dataclass
class Resultado:
    indice: int
    score: float
    texto: str


class Vetorizador:
    def __init__(self, documentos: list[str]) -> None:
        self.nome = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.modelo = None
        self.tfidf = None
        try:
            from sentence_transformers import SentenceTransformer

            self.modelo = SentenceTransformer(self.nome, local_files_only=True)
        except Exception:
            self.nome = "TF-IDF local"
            self.tfidf = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")
            self.tfidf.fit(documentos)

    def encode(self, textos: list[str]) -> np.ndarray:
        if self.modelo is not None:
            vetores = self.modelo.encode(textos, normalize_embeddings=True)
        else:
            vetores = self.tfidf.transform(textos).toarray()
        return np.asarray(vetores, dtype="float32")


class ReRanker:
    def __init__(self, documentos: list[str]) -> None:
        self.nome = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        self.modelo = None
        self.tfidf = None
        self.matriz_documentos = None
        try:
            from sentence_transformers import CrossEncoder

            self.modelo = CrossEncoder(self.nome, local_files_only=True)
        except Exception:
            try:
                from sentence_transformers import CrossEncoder

                self.nome = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                self.modelo = CrossEncoder(self.nome, local_files_only=True)
            except Exception:
                self.nome = "TF-IDF local"
                self.tfidf = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")
                self.matriz_documentos = self.tfidf.fit_transform(documentos)

    def ordenar(self, query: str, candidatos: list[Resultado]) -> list[Resultado]:
        if self.modelo is not None:
            pares = [(query, candidato.texto) for candidato in candidatos]
            scores = self.modelo.predict(pares)
        else:
            vetor_query = self.tfidf.transform([query])
            matriz = self.matriz_documentos[[candidato.indice for candidato in candidatos]]
            scores = cosine_similarity(vetor_query, matriz).ravel()
        resultados = [
            Resultado(indice=candidato.indice, score=float(score), texto=candidato.texto)
            for candidato, score in zip(candidatos, scores)
        ]
        return sorted(resultados, key=lambda item: item.score, reverse=True)


def gerar_hyde_local(query: str) -> str:
    termos = {
        "dor de cabeca": "cefaleia",
        "latejante": "pulsatil",
        "luz": "fotofobia",
        "enjoo": "nauseas",
        "chiado": "sibilancia",
        "falta de ar": "dispneia",
        "queimacao": "pirose",
        "ardor": "disuria",
        "coceira": "prurido",
        "tontura": "vertigem",
    }
    encontrados = [tecnico for popular, tecnico in termos.items() if popular in query.lower()]
    if not encontrados:
        encontrados = ["sinais clinicos correlacionados"]
    return (
        f"Paciente relata {query}. O quadro pode ser descrito tecnicamente com "
        f"{', '.join(encontrados)}, considerando sintomas associados, fatores desencadeantes "
        "e hipoteses diferenciais presentes em manual medico."
    )


def gerar_hyde(query: str, usar_openai: bool) -> str:
    if usar_openai and os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI

        cliente = OpenAI()
        resposta = cliente.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Gere um documento medico hipotetico, tecnico e curto a partir da pergunta coloquial.",
                },
                {"role": "user", "content": query},
            ],
            temperature=0.2,
        )
        return resposta.choices[0].message.content.strip()
    return gerar_hyde_local(query)


def criar_indice_hnsw(vetores: np.ndarray) -> faiss.IndexHNSWFlat:
    dimensao = vetores.shape[1]
    indice = faiss.IndexHNSWFlat(dimensao, 16, faiss.METRIC_INNER_PRODUCT)
    indice.hnsw.efConstruction = 80
    indice.hnsw.efSearch = 32
    indice.add(vetores)
    return indice


def buscar_top_10(indice: faiss.IndexHNSWFlat, vetor_query: np.ndarray) -> list[Resultado]:
    scores, ids = indice.search(vetor_query, 10)
    return [
        Resultado(indice=int(i), score=float(score), texto=DOCUMENTOS[int(i)])
        for i, score in zip(ids[0], scores[0])
        if i != -1
    ]


def imprimir_resultados(titulo: str, resultados: list[Resultado]) -> None:
    print(f"\n{titulo}")
    print("-" * len(titulo))
    for posicao, resultado in enumerate(resultados, start=1):
        texto = textwrap.fill(resultado.texto, width=92, subsequent_indent="      ")
        print(f"{posicao:02d}. score={resultado.score:.4f} | doc_id={resultado.indice}")
        print(f"    {texto}")


def executar(query: str, usar_openai: bool) -> None:
    vetorizador = Vetorizador(DOCUMENTOS)
    vetores_documentos = vetorizador.encode(DOCUMENTOS)
    faiss.normalize_L2(vetores_documentos)

    indice = criar_indice_hnsw(vetores_documentos)
    documento_hipotetico = gerar_hyde(query, usar_openai)
    vetor_hyde = vetorizador.encode([documento_hipotetico])
    faiss.normalize_L2(vetor_hyde)

    top_10 = buscar_top_10(indice, vetor_hyde)
    reranker = ReRanker(DOCUMENTOS)
    top_3 = reranker.ordenar(query, top_10)[:3]

    print(f"Query original: {query}")
    print(f"Embedding: {vetorizador.nome}")
    print(f"Cross-Encoder: {reranker.nome}")
    print("\nDocumento hipotetico HyDE:")
    print(textwrap.fill(documento_hipotetico, width=96))
    imprimir_resultados("Top-10 recuperados pelo HNSW", top_10)
    imprimir_resultados("Top-3 apos Cross-Encoder", top_3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="dor de cabeca latejante e luz incomodando")
    parser.add_argument("--use-openai-hyde", action="store_true")
    args = parser.parse_args()
    executar(args.query, args.use_openai_hyde)


if __name__ == "__main__":
    main()
