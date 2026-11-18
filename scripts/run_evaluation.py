"""
run_evaluation.py
=================
Script principal para execução completa da avaliação do all-MiniLM-L12-v2.
Executa todos os testes programaticamente, gera métricas JSON, salva predições
e renderiza todos os gráficos PNG nas pastas results/metrics/, results/plots/ e results/predictions/.

Uso:
    python scripts/run_evaluation.py                    # Roda tudo (métricas + gráficos + predições)
    python scripts/run_evaluation.py --step inspection  # Roda só a inspeção
    python scripts/run_evaluation.py --step semantic    # Roda só similaridade
    python scripts/run_evaluation.py --step retrieval   # Roda só recuperação
    python scripts/run_evaluation.py --step bias        # Roda só viés
    python scripts/run_evaluation.py --step robustness  # Roda só robustez
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Backend não interativo para geração de gráficos sem abrir janelas
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from scipy.stats import spearmanr, pearsonr

# Estilo visual padronizado
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

# ──────────────────────────────────────────────
# Configuração de Caminhos
# ──────────────────────────────────────────────

MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PLOTS_DIR = RESULTS_DIR / "plots"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

for d in [METRICS_DIR, PLOTS_DIR, PREDICTIONS_DIR, DATA_DIR / "test_sets", DATA_DIR / "processed"]:
    d.mkdir(parents=True, exist_ok=True)


def load_model(model_name: str = MODEL_NAME) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Carregando {model_name} no dispositivo: {device} ...")
    t0 = time.time()
    model = SentenceTransformer(model_name, device=device)
    print(f"Modelo carregado em {time.time() - t0:.2f}s")
    return model


# ──────────────────────────────────────────────
# Etapa 1: Inspeção Técnica
# ──────────────────────────────────────────────

def run_inspection(model: SentenceTransformer) -> dict:
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
        trunc = len(tokenizer(text, truncation=True, max_length=model.max_seq_length)["input_ids"])
        truncation_results.append({
            "text_preview": text[:40],
            "original_tokens": orig,
            "truncated_tokens": trunc,
            "tokens_dropped": max(0, orig - trunc),
            "truncated": orig > model.max_seq_length,
        })

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

    # Gráfico de exemplo de embeddings
    sample_texts = [
        "A Educação de Jovens e Adultos (EJA) é voltada para quem não concluiu os estudos.",
        "Como realizar matrícula na EJA?",
        "Qual é a idade mínima para a EJA?",
        "A avaliação da EJA é contínua e formativa.",
        "Posso solicitar o histórico escolar na secretaria."
    ]
    sample_embs = model.encode(sample_texts, normalize_embeddings=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, emb in enumerate(sample_embs):
        ax.plot(emb[:80], label=f"Sentença {i+1}", alpha=0.8)
    ax.set_title("Amostra das Primeiras 80 Dimensões dos Embeddings (all-MiniLM-L12-v2)")
    ax.set_xlabel("Dimensão do Embedding")
    ax.set_ylabel("Valor")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plot_file = PLOTS_DIR / "01_embeddings_sample.png"
    fig.savefig(plot_file)
    plt.close(fig)

    out_path = METRICS_DIR / "01_inspection_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Dimensão dos embeddings : {report['embedding_dimension']}")
    print(f"Parâmetros totais       : {report['n_parameters']:,}")
    print(f"Comprimento máximo      : {report['max_seq_length']} tokens")
    print(f"Gráfico gerado em       : {plot_file}")
    print(f"Métricas salvas em      : {out_path}")
    return report


# ──────────────────────────────────────────────
# Etapa 2: Desempenho Semântico
# ──────────────────────────────────────────────

def run_semantic(model: SentenceTransformer) -> dict:
    print("\n=== ETAPA 2: DESEMPENHO SEMÂNTICO ===")
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

    # Salva predições detalhadas
    pred_path = PREDICTIONS_DIR / "semantic_predictions.csv"
    df.to_csv(pred_path, index=False, encoding="utf-8")

    # Gráfico 1: Scatter plot
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(data=df, x="human_score", y="model_similarity", hue="category", s=80, alpha=0.85, ax=ax)
    sns.regplot(data=df, x="human_score", y="model_similarity", scatter=False, ax=ax, color="darkgray", line_kws={"linestyle": "--"})
    ax.set_title(f"Similaridade Humana vs. MiniLM (Spearman: {spearman:.3f}, Pearson: {pearson_r:.3f})")
    ax.set_xlabel("Nota Humana (1 a 5)")
    ax.set_ylabel("Similaridade do Modelo (Cosine)")
    plt.tight_layout()
    plot_scatter = PLOTS_DIR / "02_scatter_similarity.png"
    fig.savefig(plot_scatter)
    plt.close(fig)

    # Gráfico 2: Boxplot por nota humana
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(data=df, x="human_score", y="model_similarity", ax=ax, palette="Blues")
    sns.stripplot(data=df, x="human_score", y="model_similarity", ax=ax, color="black", alpha=0.5, jitter=0.2)
    ax.set_title("Distribuição da Similaridade por Nota Humana")
    ax.set_xlabel("Nota Humana")
    ax.set_ylabel("Similaridade Coseno")
    plt.tight_layout()
    plot_box = PLOTS_DIR / "02_boxplot_by_score.png"
    fig.savefig(plot_box)
    plt.close(fig)

    report = {
        "n_pairs": len(df),
        "spearman": spearman,
        "pearson": pearson_r,
        "mean_similarity": float(df["model_similarity"].mean()),
        "std_similarity": float(df["model_similarity"].std()),
    }

    out_path = METRICS_DIR / "02_semantic_performance.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Pares avaliados   : {report['n_pairs']}")
    print(f"Spearman          : {report['spearman']:.4f}")
    print(f"Pearson           : {report['pearson']:.4f}")
    print(f"Predições salvas  : {pred_path}")
    print(f"Gráficos salvos   : {plot_scatter}, {plot_box}")
    print(f"Métricas salvas em: {out_path}")
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
    print("\n=== ETAPA 3: RECUPERAÇÃO DE INFORMAÇÃO ===")
    corpus_path = DATA_DIR / "test_sets" / "corpus.json"
    queries_path = DATA_DIR / "test_sets" / "queries.json"
    relevant_path = DATA_DIR / "test_sets" / "relevant_docs.json"

    for p in [corpus_path, queries_path, relevant_path]:
        if not p.exists():
            print(f"⚠ Arquivo não encontrado: {p}")
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
    prediction_rows = []
    for i, qid in enumerate(query_ids):
        order = np.argsort(sim_matrix[i])[::-1]
        ranking = [corpus_ids[j] for j in order]
        rel = relevant_docs.get(qid, set())
        
        # Registra predições top-5 para salvamento
        for rank_pos, doc_idx in enumerate(order[:5], start=1):
            doc_id = corpus_ids[doc_idx]
            prediction_rows.append({
                "query_id": qid,
                "query_text": queries[qid],
                "rank": rank_pos,
                "retrieved_doc_id": doc_id,
                "similarity_score": float(sim_matrix[i][doc_idx]),
                "is_relevant": doc_id in rel
            })

        per_query.append({
            "query_id": qid,
            "recall@1": recall_at_k(ranking, rel, 1),
            "recall@3": recall_at_k(ranking, rel, 3),
            "recall@5": recall_at_k(ranking, rel, 5),
            "recall@10": recall_at_k(ranking, rel, 10),
            "precision@5": precision_at_k(ranking, rel, 5),
            "mrr@10": mrr(ranking, rel),
            "ndcg@10": ndcg_at_k(ranking, rel, 10),
        })

    # Salva predições brutas de ranking
    pred_df = pd.DataFrame(prediction_rows)
    pred_path = PREDICTIONS_DIR / "retrieval_rankings.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8")

    pq_df = pd.DataFrame(per_query)
    report = {
        "n_queries": len(query_ids),
        "n_docs": len(corpus_ids),
        "recall@1": float(pq_df["recall@1"].mean()),
        "recall@3": float(pq_df["recall@3"].mean()),
        "recall@5": float(pq_df["recall@5"].mean()),
        "recall@10": float(pq_df["recall@10"].mean()),
        "precision@5": float(pq_df["precision@5"].mean()),
        "mrr@10": float(pq_df["mrr@10"].mean()),
        "ndcg@10": float(pq_df["ndcg@10"].mean()),
    }

    # Gráfico 1: Recall em função de K
    k_vals = ["Recall@1", "Recall@3", "Recall@5", "Recall@10"]
    r_vals = [report["recall@1"], report["recall@3"], report["recall@5"], report["recall@10"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=k_vals, y=r_vals, ax=ax, palette="Blues_d")
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(r_vals):
        ax.text(i, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")
    ax.set_title("Recall em Diferentes Cortes K (all-MiniLM-L12-v2)")
    ax.set_ylabel("Recall")
    plt.tight_layout()
    plot_recall = PLOTS_DIR / "03_recall_at_k.png"
    fig.savefig(plot_recall)
    plt.close(fig)

    # Gráfico 2: Métricas principais de IR
    metrics_names = ["MRR@10", "nDCG@10", "Precision@5", "Recall@10"]
    metrics_vals = [report["mrr@10"], report["ndcg@10"], report["precision@5"], report["recall@10"]]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    sns.barplot(x=metrics_names, y=metrics_vals, ax=ax, palette="crest")
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(metrics_vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    ax.set_title("Resumo das Métricas de Recuperação de Informação")
    plt.tight_layout()
    plot_metrics = PLOTS_DIR / "03_retrieval_metrics.png"
    fig.savefig(plot_metrics)
    plt.close(fig)

    out_path = METRICS_DIR / "03_retrieval_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Recall@5          : {report['recall@5']:.4f}")
    print(f"Recall@10         : {report['recall@10']:.4f}")
    print(f"MRR@10            : {report['mrr@10']:.4f}")
    print(f"nDCG@10           : {report['ndcg@10']:.4f}")
    print(f"Predições salvas  : {pred_path}")
    print(f"Gráficos salvos   : {plot_recall}, {plot_metrics}")
    print(f"Métricas salvas em: {out_path}")
    return report


# ──────────────────────────────────────────────
# Etapa 4: Viés
# ──────────────────────────────────────────────

def run_bias(model: SentenceTransformer) -> dict:
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
    bias_rows = []
    for a, b in gender_pairs:
        emb = model.encode([a, b], normalize_embeddings=True)
        s = float(np.dot(emb[0], emb[1]))
        sims.append(s)
        bias_rows.append({"frase_a": a, "frase_b": b, "similaridade": s, "tipo": "contrafactual_genero"})

    sims_arr = np.array(sims)
    boot_means = [np.random.choice(sims_arr, size=len(sims_arr), replace=True).mean() for _ in range(5000)]
    ci_low, ci_high = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))

    # Salva predições
    pred_path = PREDICTIONS_DIR / "bias_predictions.csv"
    pd.DataFrame(bias_rows).to_csv(pred_path, index=False, encoding="utf-8")

    # Gráfico de distribuição de similaridade contrafactual
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(sims_arr, bins=8, kde=True, ax=ax, color="purple")
    ax.axvline(sims_arr.mean(), color="red", linestyle="--", label=f"Média: {sims_arr.mean():.3f}")
    ax.set_title("Distribuição de Similaridade Contrafactual de Gênero")
    ax.set_xlabel("Similaridade Coseno")
    ax.legend()
    plt.tight_layout()
    plot_gender = PLOTS_DIR / "04_gender_similarity_dist.png"
    fig.savefig(plot_gender)
    plt.close(fig)

    report = {
        "gender_counterfactual": {
            "n_pairs": len(gender_pairs),
            "mean_similarity": float(sims_arr.mean()),
            "median_similarity": float(np.median(sims_arr)),
            "std_similarity": float(sims_arr.std()),
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
        }
    }

    out_path = METRICS_DIR / "04_bias_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Similaridade média de gênero: {report['gender_counterfactual']['mean_similarity']:.4f}")
    print(f"IC 95%                      : [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"Gráfico gerado em           : {plot_gender}")
    print(f"Métricas salvas em          : {out_path}")
    return report


# ──────────────────────────────────────────────
# Etapa 5: Robustez
# ──────────────────────────────────────────────

def run_robustness(model: SentenceTransformer) -> dict:
    print("\n=== ETAPA 5: ROBUSTEZ ===")
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

    neg_rows = []
    for a, b in negation_pairs:
        emb = model.encode([a, b], normalize_embeddings=True)
        s = float(np.dot(emb[0], emb[1]))
        neg_rows.append({"frase_afirmativa": a, "frase_negativa": b, "similaridade": s})

    neg_df = pd.DataFrame(neg_rows)
    neg_sims = neg_df["similaridade"].values

    pred_path = PREDICTIONS_DIR / "robustness_predictions.csv"
    neg_df.to_csv(pred_path, index=False, encoding="utf-8")

    # Gráfico de Negação
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=neg_df, x=neg_df.index, y="similaridade", ax=ax, palette="Reds_d")
    ax.axhline(0.85, color="darkred", linestyle="--", label="Zona de Risco RAG (> 0.85)")
    ax.set_title("Similaridade Coseno em Pares Afirmativo vs. Negativo")
    ax.set_xlabel("Índice do Par")
    ax.set_ylabel("Similaridade Coseno")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plot_neg = PLOTS_DIR / "05_negation_similarity.png"
    fig.savefig(plot_neg)
    plt.close(fig)

    report = {
        "negation": {
            "n_pairs": len(negation_pairs),
            "mean_similarity": float(neg_sims.mean()),
            "std_similarity": float(neg_sims.std()),
            "high_similarity_pairs": int((neg_sims > 0.85).sum()),
            "risk": "ALTO" if neg_sims.mean() > 0.85 else "MÉDIO",
        }
    }

    out_path = METRICS_DIR / "05_robustness_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Negação — similaridade média: {report['negation']['mean_similarity']:.4f} (Risco: {report['negation']['risk']})")
    print(f"Gráfico gerado em           : {plot_neg}")
    print(f"Métricas salvas em          : {out_path}")
    return report


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
    parser = argparse.ArgumentParser(description="Executa a avaliação do all-MiniLM-L12-v2.")
    parser.add_argument("--step", choices=list(STEPS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    t_init = time.time()
    model = load_model()

    if args.step == "all":
        for name, fn in STEPS.items():
            fn(model)
    else:
        STEPS[args.step](model)

    print(f"\n==========================================")
    print(f"✓ Execução finalizada em {time.time() - t_init:.2f}s")
    print(f"  Métricas    : {METRICS_DIR}")
    print(f"  Gráficos    : {PLOTS_DIR}")
    print(f"  Predições   : {PREDICTIONS_DIR}")
    print(f"==========================================")
