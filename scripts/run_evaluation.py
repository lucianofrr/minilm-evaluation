"""
run_evaluation.py
=================
Script principal para execução de avaliação do MiniLM (com suporte a modelos multilíngues e comparação).

Modelos suportados nativamente:
  - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Recomendado para Português)
  - sentence-transformers/all-MiniLM-L12-v2 (Original em Inglês)

Uso:
    python scripts/run_evaluation.py                                     # Roda o paraphrase-multilingual-MiniLM-L12-v2
    python scripts/run_evaluation.py --model sentence-transformers/all-MiniLM-L12-v2
    python scripts/run_evaluation.py --compare                           # Compara os dois modelos lado a lado
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr, pearsonr

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

for d in [METRICS_DIR, PLOTS_DIR, PREDICTIONS_DIR, DATA_DIR / "test_sets", DATA_DIR / "processed"]:
    d.mkdir(parents=True, exist_ok=True)


def load_model(model_name: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Carregando {model_name} no dispositivo: {device} ...")
    t0 = time.time()
    model = SentenceTransformer(model_name, device=device)
    print(f"Modelo carregado em {time.time() - t0:.2f}s")
    return model


def get_model_slug(model_name: str) -> str:
    return "multilingual-MiniLM" if "multilingual" in model_name.lower() else "all-MiniLM"


# ──────────────────────────────────────────────
# Etapa 1: Inspeção Técnica
# ──────────────────────────────────────────────

def run_inspection(model: SentenceTransformer, model_name: str) -> dict:
    slug = get_model_slug(model_name)
    print(f"\n=== ETAPA 1: INSPEÇÃO TÉCNICA [{slug}] ===")
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    n_params = sum(p.numel() for p in base_model.parameters())
    config = base_model.config

    texts_truncation = [
        "Texto curto para teste.",
        " ".join(["palavra"] * 150),
        " ".join(["palavra"] * 300),
        " ".join(["palavra"] * 500),
    ]
    truncation_results = []
    for text in texts_truncation:
        orig = len(tokenizer(text, truncation=False)["input_ids"])
        trunc = len(tokenizer(text, truncation=True, max_length=model.max_seq_length)["input_ids"])
        truncation_results.append({
            "text_preview": text[:40],
            "original_tokens": orig,
            "truncated_tokens": trunc,
            "tokens_dropped": max(0, orig - trunc),
            "truncated": orig > model.max_seq_length,
        })

    report = {
        "model_name": model_name,
        "embedding_dimension": model.get_sentence_embedding_dimension(),
        "max_seq_length": model.max_seq_length,
        "similarity_fn": model.similarity_fn_name,
        "n_parameters": n_params,
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "hidden_size": getattr(config, "hidden_size", None),
        "vocab_size": tokenizer.vocab_size,
        "truncation_tests": truncation_results,
    }

    out_path = METRICS_DIR / f"01_inspection_report_{slug}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Dimensão dos embeddings : {report['embedding_dimension']}")
    print(f"Parâmetros totais       : {report['n_parameters']:,}")
    print(f"Vocabulário Tokenizer   : {report['vocab_size']:,} tokens")
    print(f"Comprimento máximo      : {report['max_seq_length']} tokens")
    print(f"Métricas salvas em      : {out_path}")
    return report


# ──────────────────────────────────────────────
# Etapa 2: Desempenho Semântico
# ──────────────────────────────────────────────

def run_semantic(model: SentenceTransformer, model_name: str) -> dict:
    slug = get_model_slug(model_name)
    print(f"\n=== ETAPA 2: DESEMPENHO SEMÂNTICO [{slug}] ===")
    csv_path = DATA_DIR / "test_sets" / "semantic_pairs.csv"
    if not csv_path.exists():
        print(f"⚠ Dataset não encontrado: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)
    emb_a = model.encode(df["sentence_a"].tolist(), normalize_embeddings=True)
    emb_b = model.encode(df["sentence_b"].tolist(), normalize_embeddings=True)
    df["model_similarity"] = np.sum(emb_a * emb_b, axis=1)

    spearman = float(spearmanr(df["human_score"], df["model_similarity"]).statistic)
    pearson_r = float(pearsonr(df["human_score"], df["model_similarity"])[0])

    pred_path = PREDICTIONS_DIR / f"semantic_predictions_{slug}.csv"
    df.to_csv(pred_path, index=False, encoding="utf-8")

    # Gráfico de Dispersão
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="human_score", y="model_similarity", hue="category", s=80, alpha=0.85, ax=ax)
    sns.regplot(data=df, x="human_score", y="model_similarity", scatter=False, ax=ax, color="darkgray", line_kws={"linestyle": "--"})
    ax.set_title(f"Similaridade Humana vs. {slug}\n(Spearman: {spearman:.3f}, Pearson: {pearson_r:.3f})")
    ax.set_xlabel("Nota Humana (1 a 5)")
    ax.set_ylabel("Similaridade do Modelo (Cosine)")
    plt.tight_layout()
    plot_scatter = PLOTS_DIR / f"02_scatter_similarity_{slug}.png"
    fig.savefig(plot_scatter)
    plt.close(fig)

    report = {
        "model": model_name,
        "n_pairs": len(df),
        "spearman": spearman,
        "pearson": pearson_r,
        "mean_similarity": float(df["model_similarity"].mean()),
        "std_similarity": float(df["model_similarity"].std()),
    }

    out_path = METRICS_DIR / f"02_semantic_performance_{slug}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Spearman          : {report['spearman']:.4f}")
    print(f"Pearson           : {report['pearson']:.4f}")
    print(f"Gráfico salvo em  : {plot_scatter}")
    print(f"Métricas salvas em: {out_path}")
    return report


# ──────────────────────────────────────────────
# Etapa 3: Recuperação de Informação
# ──────────────────────────────────────────────

def recall_at_k(ranking, relevant, k):
    return len(set(ranking[:k]) & relevant) / len(relevant) if relevant else 0.0

def mrr(ranking, relevant):
    for i, doc in enumerate(ranking, 1):
        if doc in relevant:
            return 1.0 / i
    return 0.0

def ndcg_at_k(ranking, relevant, k):
    dcg = sum(1 / np.log2(i + 2) for i, d in enumerate(ranking[:k]) if d in relevant)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def run_retrieval(model: SentenceTransformer, model_name: str) -> dict:
    slug = get_model_slug(model_name)
    print(f"\n=== ETAPA 3: RECUPERAÇÃO DE INFORMAÇÃO [{slug}] ===")
    corpus_path = DATA_DIR / "test_sets" / "corpus.json"
    queries_path = DATA_DIR / "test_sets" / "queries.json"
    relevant_path = DATA_DIR / "test_sets" / "relevant_docs.json"

    with open(corpus_path, encoding="utf-8") as f:
        corpus = json.load(f)
    with open(queries_path, encoding="utf-8") as f:
        queries = json.load(f)
    with open(relevant_path, encoding="utf-8") as f:
        relevant_docs = {q: set(v) for q, v in json.load(f).items()}

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
        per_query.append({
            "query_id": qid,
            "recall@1": recall_at_k(ranking, rel, 1),
            "recall@5": recall_at_k(ranking, rel, 5),
            "recall@10": recall_at_k(ranking, rel, 10),
            "mrr@10": mrr(ranking, rel),
            "ndcg@10": ndcg_at_k(ranking, rel, 10),
        })

    pq_df = pd.DataFrame(per_query)
    report = {
        "model": model_name,
        "n_queries": len(query_ids),
        "n_docs": len(corpus_ids),
        "recall@1": float(pq_df["recall@1"].mean()),
        "recall@5": float(pq_df["recall@5"].mean()),
        "recall@10": float(pq_df["recall@10"].mean()),
        "mrr@10": float(pq_df["mrr@10"].mean()),
        "ndcg@10": float(pq_df["ndcg@10"].mean()),
    }

    out_path = METRICS_DIR / f"03_retrieval_metrics_{slug}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Recall@1          : {report['recall@1']:.4f}")
    print(f"Recall@5          : {report['recall@5']:.4f}")
    print(f"Recall@10         : {report['recall@10']:.4f}")
    print(f"MRR@10            : {report['mrr@10']:.4f}")
    print(f"Métricas salvas em: {out_path}")
    return report


# ──────────────────────────────────────────────
# Etapa 4: Robustez (Negação)
# ──────────────────────────────────────────────

def run_robustness(model: SentenceTransformer, model_name: str) -> dict:
    slug = get_model_slug(model_name)
    print(f"\n=== ETAPA 4: ROBUSTEZ À NEGAÇÃO [{slug}] ===")
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

    neg_sims = np.array(neg_sims)
    mean_neg = float(neg_sims.mean())
    risk = "ALTO (Inseguro para RAG)" if mean_neg > 0.85 else "MODERADO" if mean_neg > 0.65 else "BAIXO (Seguro)"

    report = {
        "model": model_name,
        "n_pairs": len(negation_pairs),
        "mean_similarity": mean_neg,
        "std_similarity": float(neg_sims.std()),
        "risk": risk,
    }

    out_path = METRICS_DIR / f"05_robustness_results_{slug}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Similaridade média de negação: {mean_neg:.4f} | Risco: {risk}")
    print(f"Métricas salvas em           : {out_path}")
    return report


# ──────────────────────────────────────────────
# Comparação Direta Lado a Lado
# ──────────────────────────────────────────────

def run_comparison():
    print("\n" + "="*65)
    print("   BENCHMARK COMPARATIVO: all-MiniLM vs paraphrase-multilingual")
    print("="*65)

    models_to_test = [
        ("all-MiniLM-L12-v2", "sentence-transformers/all-MiniLM-L12-v2"),
        ("multilingual-MiniLM-L12-v2", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    ]

    records = []
    for label, m_id in models_to_test:
        print(f"\n--- Avaliando: {label} ---")
        m = load_model(m_id)
        r_sem = run_semantic(m, m_id)
        r_ret = run_retrieval(m, m_id)
        r_rob = run_robustness(m, m_id)
        records.append({
            "Modelo": label,
            "Spearman ρ": r_sem["spearman"],
            "Pearson r": r_sem["pearson"],
            "Recall@1": r_ret["recall@1"],
            "Recall@5": r_ret["recall@5"],
            "Recall@10": r_ret["recall@10"],
            "MRR@10": r_ret["mrr@10"],
            "Sim. Negação": r_rob["mean_similarity"],
        })

    comp_df = pd.DataFrame(records)
    print("\n" + "="*65)
    print("TABELA CONSOLIDADA DE COMPARAÇÃO:")
    print("="*65)
    print(comp_df.to_string(index=False))

    # Gráfico Comparativo
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # 1. Semântica e Negação
    metrics_a = ["Spearman ρ", "Pearson r", "Sim. Negação"]
    melted_a = comp_df.melt(id_vars="Modelo", value_vars=metrics_a, var_name="Métrica", value_name="Score")
    sns.barplot(data=melted_a, x="Métrica", y="Score", hue="Modelo", ax=axes[0], palette="Set2")
    axes[0].set_title("Correlação Semântica e Sensibilidade à Negação\n(Negação menor é melhor)")
    axes[0].set_ylim(0, 1.05)

    # 2. Recuperação de Informação (IR)
    metrics_b = ["Recall@1", "Recall@5", "Recall@10", "MRR@10"]
    melted_b = comp_df.melt(id_vars="Modelo", value_vars=metrics_b, var_name="Métrica", value_name="Score")
    sns.barplot(data=melted_b, x="Métrica", y="Score", hue="Modelo", ax=axes[1], palette="crest")
    axes[1].set_title("Métricas de Recuperação (Corpus EJA)")
    axes[1].set_ylim(0, 1.05)

    plt.tight_layout()
    comp_plot = PLOTS_DIR / "comparison_all_vs_multilingual_minilm.png"
    fig.savefig(comp_plot)
    plt.close(fig)
    print(f"\nGráfico comparativo salvo em: {comp_plot}")


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação experimental do MiniLM.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="ID do modelo no HuggingFace.")
    parser.add_argument("--compare", action="store_true", help="Executa comparação direta entre all-MiniLM e multilingual-MiniLM.")
    args = parser.parse_args()

    if args.compare:
        run_comparison()
    else:
        m = load_model(args.model)
        run_inspection(m, args.model)
        run_semantic(m, args.model)
        run_retrieval(m, args.model)
        run_robustness(m, args.model)
