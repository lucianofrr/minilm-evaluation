"""
run_evaluation.py
=================
Script principal para execução completa da avaliação do all-MiniLM-L12-v2.
Executa todos os testes programaticamente (sem Jupyter) e salva resultados.

Uso:
    python scripts/run_evaluation.py                    # Roda tudo
    python scripts/run_evaluation.py --step inspection  # Roda só a inspeção
    python scripts/run_evaluation.py --step semantic    # Roda só similaridade
    python scripts/run_evaluation.py --step retrieval   # Roda só recuperação
    python scripts/run_evaluation.py --step bias        # Roda só viés
    python scripts/run_evaluation.py --step robustness  # Roda só robustez
    python scripts/run_evaluation.py --step reliability # Roda só confiabilidade
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────
# Configuração
# ──────────────────────────────────────────────

MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"

# Garante que os diretórios existem
for d in [METRICS_DIR, PLOTS_DIR, DATA_DIR / "test_sets", DATA_DIR / "processed"]:
    d.mkdir(parents=True, exist_ok=True)


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Carregando {model_name} no dispositivo: {device}")
    model = SentenceTransformer(model_name, device=device)
    return model


# ──────────────────────────────────────────────
# Etapa 1: Inspeção Técnica
# ──────────────────────────────────────────────

def run_inspection(model: SentenceTransformer) -> dict:
    """Registra arquitetura, parâmetros, pooling e truncamento."""
    print("\n=== ETAPA 1: INSPEÇÃO TÉCNICA ===")

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModel.from_pretrained(MODEL_NAME)

    n_params = sum(p.numel() for p in base_model.parameters())
    config = base_model.config

    # Teste de truncamento
    texts_truncation = [
        "Texto curto para teste.",
        " ".join(["palavra"] * 150),
        " ".join(["palavra"] * 300),
        " ".join(["palavra"] * 500),
    ]
    truncation_results = []
    for text in texts_truncation:
        orig = len(tokenizer(text, truncation=False)["input_ids"])
        trunc = len(
            tokenizer(text, truncation=True, max_length=model.max_seq_length)[
                "input_ids"
            ]
        )
        truncation_results.append(
            {
                "text_preview": text[:40],
                "original_tokens": orig,
                "truncated_tokens": trunc,
                "tokens_dropped": max(0, orig - trunc),
                "truncated": orig > model.max_seq_length,
            }
        )

    report = {
        "model_name": MODEL_NAME,
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "max_seq_length": model.max_seq_length,
        "similarity_fn": model.similarity_fn_name,
        "n_parameters": n_params,
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "vocab_size": config.vocab_size,
        "truncation_tests": truncation_results,
    }

    out_path = METRICS_DIR / "01_inspection_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Dimensão dos embeddings : {report['embedding_dimension']}")
    print(f"Parâmetros totais       : {report['n_parameters']:,}")
    print(f"Camadas                 : {report['num_hidden_layers']}")
    print(f"Comprimento máximo      : {report['max_seq_length']} tokens")
    print(f"Resultados salvos em    : {out_path}")
    return report


# ──────────────────────────────────────────────
# Etapa 2: Desempenho Semântico
# ──────────────────────────────────────────────

def run_semantic(model: SentenceTransformer) -> dict:
    """Calcula Spearman/Pearson sobre o dataset de pares."""
    print("\n=== ETAPA 2: DESEMPENHO SEMÂNTICO ===")

    csv_path = DATA_DIR / "test_sets" / "semantic_pairs.csv"
    if not csv_path.exists():
        print(f"⚠  Dataset não encontrado: {csv_path}")
        print("   Execute o notebook 02_semantic_performance.ipynb primeiro para gerar os dados.")
        return {}

    df = pd.read_csv(csv_path)
    emb_a = model.encode(df["sentence_a"].tolist(), normalize_embeddings=True)
    emb_b = model.encode(df["sentence_b"].tolist(), normalize_embeddings=True)
    df["model_similarity"] = np.sum(emb_a * emb_b, axis=1)

    spearman = spearmanr(df["human_score"], df["model_similarity"]).statistic
    pearson_r = pearsonr(df["human_score"], df["model_similarity"])[0]

    # Acurácia de trincas (gera trincas automaticamente do dataset)
    high = df[df["human_score"] >= 4].reset_index(drop=True)
    low = df[df["human_score"] <= 2].reset_index(drop=True)
    n_triplets = min(len(high), len(low))
    triplet_acc = None
    if n_triplets > 0:
        a_embs = model.encode(high["sentence_a"][:n_triplets].tolist(), normalize_embeddings=True)
        p_embs = model.encode(high["sentence_b"][:n_triplets].tolist(), normalize_embeddings=True)
        n_embs = model.encode(low["sentence_b"][:n_triplets].tolist(), normalize_embeddings=True)
        pos_scores = np.sum(a_embs * p_embs, axis=1)
        neg_scores = np.sum(a_embs * n_embs, axis=1)
        triplet_acc = float(np.mean(pos_scores > neg_scores))

    report = {
        "n_pairs": len(df),
        "spearman": float(spearman),
        "pearson": float(pearson_r),
        "triplet_accuracy": triplet_acc,
        "mean_similarity": float(df["model_similarity"].mean()),
        "std_similarity": float(df["model_similarity"].std()),
    }

    out_path = METRICS_DIR / "02_semantic_performance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Pares avaliados : {report['n_pairs']}")
    print(f"Spearman        : {report['spearman']:.4f}")
    print(f"Pearson         : {report['pearson']:.4f}")
    if triplet_acc is not None:
        print(f"Acurácia trincas: {report['triplet_accuracy']:.4f}")
    print(f"Resultados salvos em: {out_path}")
    return report


# ──────────────────────────────────────────────
# Etapa 3: Recuperação de Informação
# ──────────────────────────────────────────────

def recall_at_k(ranking, relevant, k):
    return len(set(ranking[:k]) & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(ranking, relevant, k):
    return sum(1 for d in ranking[:k] if d in relevant) / k


def mrr(ranking, relevant):
    for i, doc in enumerate(ranking, 1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranking, relevant, k):
    dcg = sum(1 / np.log2(i + 2) for i, d in enumerate(ranking[:k]) if d in relevant)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def run_retrieval(model: SentenceTransformer) -> dict:
    """Avalia recuperação de documentos."""
    print("\n=== ETAPA 3: RECUPERAÇÃO DE INFORMAÇÃO ===")

    corpus_path = DATA_DIR / "test_sets" / "corpus.json"
    queries_path = DATA_DIR / "test_sets" / "queries.json"
    relevant_path = DATA_DIR / "test_sets" / "relevant_docs.json"

    for p in [corpus_path, queries_path, relevant_path]:
        if not p.exists():
            print(f"⚠  Arquivo não encontrado: {p}")
            print("   Execute o notebook 03_retrieval.ipynb primeiro.")
            return {}

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    with open(queries_path, encoding="utf-8") as f:
        queries = json.load(f)
    with open(relevant_path, encoding="utf-8") as f:
        relevant_docs_raw = json.load(f)
    relevant_docs = {q: set(v) for q, v in relevant_docs_raw.items()}

    corpus_ids = list(corpus.keys())
    corpus_texts = list(corpus.values())
    query_ids = list(queries.keys())
    query_texts = list(queries.values())

    corpus_embs = model.encode(corpus_texts, normalize_embeddings=True)
    query_embs = model.encode(query_texts, normalize_embeddings=True)
    sim_matrix = query_embs @ corpus_embs.T

    per_query = []
    for i, qid in enumerate(query_ids):
        order = np.argsort(sim_matrix[i])[::-1]
        ranking = [corpus_ids[j] for j in order]
        rel = relevant_docs.get(qid, set())
        per_query.append(
            {
                "query_id": qid,
                "recall@5": recall_at_k(ranking, rel, 5),
                "recall@10": recall_at_k(ranking, rel, 10),
                "precision@5": precision_at_k(ranking, rel, 5),
                "mrr@10": mrr(ranking, rel),
                "ndcg@10": ndcg_at_k(ranking, rel, 10),
            }
        )

    pq_df = pd.DataFrame(per_query)
    report = {
        "n_queries": len(query_ids),
        "n_docs": len(corpus_ids),
        "recall@5": float(pq_df["recall@5"].mean()),
        "recall@10": float(pq_df["recall@10"].mean()),
        "precision@5": float(pq_df["precision@5"].mean()),
        "mrr@10": float(pq_df["mrr@10"].mean()),
        "ndcg@10": float(pq_df["ndcg@10"].mean()),
        "per_query": per_query,
    }

    out_path = METRICS_DIR / "03_retrieval_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Consultas : {report['n_queries']}")
    print(f"Documentos: {report['n_docs']}")
    print(f"Recall@5  : {report['recall@5']:.4f}")
    print(f"Recall@10 : {report['recall@10']:.4f}")
    print(f"MRR@10    : {report['mrr@10']:.4f}")
    print(f"nDCG@10   : {report['ndcg@10']:.4f}")
    print(f"Resultados salvos em: {out_path}")
    return report


# ──────────────────────────────────────────────
# Etapa 4: Viés
# ──────────────────────────────────────────────

def run_bias(model: SentenceTransformer) -> dict:
    """Executa testes contrafactuais de gênero."""
    print("\n=== ETAPA 4: VIÉS ===")

    gender_pairs = [
        ("O professor explicou o conteúdo para a turma.", "A professora explicou o conteúdo para a turma."),
        ("O aluno precisa entregar a atividade até sexta.", "A aluna precisa entregar a atividade até sexta."),
        ("O estudante concluiu o curso com aprovação.", "A estudante concluiu o curso com aprovação."),
        ("O diretor convocou uma reunião de pais.", "A diretora convocou uma reunião de pais."),
        ("O coordenador aprovou o pedido de transferência.", "A coordenadora aprovou o pedido de transferência."),
        ("O aluno foi aprovado no exame final.", "A aluna foi aprovada no exame final."),
        ("O estudante solicitou o histórico escolar.", "A estudante solicitou o histórico escolar."),
        ("O jovem quer terminar o ensino médio pela EJA.", "A jovem quer terminar o ensino médio pela EJA."),
        ("O adulto voltou a estudar depois de anos afastado.", "A adulta voltou a estudar depois de anos afastada."),
        ("O funcionário pediu licença para estudar.", "A funcionária pediu licença para estudar."),
    ]

    sims = []
    for a, b in gender_pairs:
        emb = model.encode([a, b], normalize_embeddings=True)
        s = float(np.dot(emb[0], emb[1]))
        sims.append(s)

    sims_arr = np.array(sims)
    # Bootstrap CI
    boot_means = [np.random.choice(sims_arr, size=len(sims_arr), replace=True).mean() for _ in range(5000)]
    ci_low, ci_high = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    report = {
        "gender_counterfactual": {
            "n_pairs": len(gender_pairs),
            "mean_similarity": float(sims_arr.mean()),
            "median_similarity": float(np.median(sims_arr)),
            "std_similarity": float(sims_arr.std()),
            "min_similarity": float(sims_arr.min()),
            "max_similarity": float(sims_arr.max()),
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "note": "Alta similaridade indica que a troca de gênero tem pouco impacto na representação.",
        }
    }

    out_path = METRICS_DIR / "04_bias_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Pares de gênero testados : {report['gender_counterfactual']['n_pairs']}")
    print(f"Similaridade média       : {report['gender_counterfactual']['mean_similarity']:.4f}")
    print(f"IC 95%                   : [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Resultados salvos em: {out_path}")
    return report


# ──────────────────────────────────────────────
# Etapa 5: Robustez
# ──────────────────────────────────────────────

def run_robustness(model: SentenceTransformer) -> dict:
    """Testa negação, erros ortográficos e paráfrases."""
    print("\n=== ETAPA 5: ROBUSTEZ ===")

    # Negação
    negation_pairs = [
        ("O estudante pode realizar a matrícula.", "O estudante não pode realizar a matrícula."),
        ("A escola oferece ensino gratuito.", "A escola não oferece ensino gratuito."),
        ("O aluno concluiu o ensino médio.", "O aluno não concluiu o ensino médio."),
        ("A EJA aceita transferência de outra escola.", "A EJA não aceita transferência de outra escola."),
        ("Preciso de documentos para me matricular.", "Não preciso de documentos para me matricular."),
        ("O aluno foi aprovado.", "O aluno não foi aprovado."),
        ("A escola oferece transporte escolar.", "A escola não oferece transporte escolar."),
        ("O candidato tem direito à vaga.", "O candidato não tem direito à vaga."),
        ("A matrícula está aberta.", "A matrícula não está aberta."),
        ("O certificado é reconhecido pelo MEC.", "O certificado não é reconhecido pelo MEC."),
    ]

    neg_sims = []
    for a, b in negation_pairs:
        emb = model.encode([a, b], normalize_embeddings=True)
        neg_sims.append(float(np.dot(emb[0], emb[1])))

    neg_arr = np.array(neg_sims)

    # Ortografia
    ortho_groups = [
        ("Como funciona a Educação de Jovens e Adultos?", "Como funciona a Educacao de Jovens e Adultos?"),
        ("Preciso de um histórico escolar.", "Preciso de um historico escolar."),
        ("Qual é a idade mínima para a EJA?", "Qual e a idade minima para a EJA?"),
        ("A matrícula é gratuita.", "A matricula e gratuita."),
        ("Posso me inscrever no período noturno?", "Posso me inscrever no periodo noturno?"),
    ]
    ortho_sims = []
    for a, b in ortho_groups:
        emb = model.encode([a, b], normalize_embeddings=True)
        ortho_sims.append(float(np.dot(emb[0], emb[1])))

    ortho_arr = np.array(ortho_sims)

    report = {
        "negation": {
            "n_pairs": len(negation_pairs),
            "mean_similarity": float(neg_arr.mean()),
            "std_similarity": float(neg_arr.std()),
            "high_similarity_pairs": int((neg_arr > 0.85).sum()),
            "risk": "ALTO" if neg_arr.mean() > 0.85 else "MÉDIO" if neg_arr.mean() > 0.70 else "BAIXO",
            "interpretation": (
                "Similaridade alta em pares de negação indica que o modelo não distingue "
                "bem afirmações de suas negações — risco em RAG."
            ),
        },
        "orthography": {
            "n_pairs": len(ortho_groups),
            "mean_similarity": float(ortho_arr.mean()),
            "std_similarity": float(ortho_arr.std()),
            "risk": "BAIXO" if ortho_arr.mean() > 0.85 else "MÉDIO",
        },
    }

    out_path = METRICS_DIR / "05_robustness_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Negação — similaridade média : {report['negation']['mean_similarity']:.4f} | Risco: {report['negation']['risk']}")
    print(f"Ortografia — similaridade média: {report['orthography']['mean_similarity']:.4f} | Risco: {report['orthography']['risk']}")
    print(f"Resultados salvos em: {out_path}")
    return report


# ──────────────────────────────────────────────
# Etapa 6: Relatório Consolidado
# ──────────────────────────────────────────────

def run_summary() -> None:
    """Consolida todos os resultados em uma tabela final."""
    print("\n=== RELATÓRIO CONSOLIDADO ===")

    rows = []
    metric_files = list(METRICS_DIR.glob("0*.json"))
    for mf in sorted(metric_files):
        with open(mf, encoding="utf-8") as f:
            data = json.load(f)
        rows.append({"arquivo": mf.name, "chaves": list(data.keys())[:6]})

    if rows:
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))

    print(f"\nTodos os arquivos de métricas estão em: {METRICS_DIR}")


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

STEPS = {
    "inspection": run_inspection,
    "semantic": run_semantic,
    "retrieval": run_retrieval,
    "bias": run_bias,
    "robustness": run_robustness,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Executa a avaliação completa do all-MiniLM-L12-v2."
    )
    parser.add_argument(
        "--step",
        choices=list(STEPS.keys()) + ["all"],
        default="all",
        help="Etapa a executar (padrão: all).",
    )
    args = parser.parse_args()

    t0 = time.time()
    model = load_model()

    if args.step == "all":
        for name, fn in STEPS.items():
            fn(model)
        run_summary()
    else:
        STEPS[args.step](model)

    print(f"\n✓ Concluído em {time.time() - t0:.1f}s")
