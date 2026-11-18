# Avaliação do all-MiniLM-L12-v2

Projeto de avaliação experimental, documentada e reproduzível do modelo [`sentence-transformers/all-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) para aplicações de recuperação semântica em **português brasileiro**, com foco em RAG educacional (Educação de Jovens e Adultos — EJA).

---

## Objetivo

Responder a cinco perguntas centrais:

| # | Dimensão | Pergunta |
|---|----------|----------|
| 1 | **Viés** | O modelo trata de maneira diferente frases equivalentes quando mudamos gênero, raça ou contexto socioeconômico? |
| 2 | **Desempenho** | Ele identifica corretamente textos semanticamente relacionados? |
| 3 | **Robustez** | Continua funcionando diante de negações, erros de escrita, paráfrases e mudanças de domínio? |
| 4 | **Confiabilidade** | Os resultados são estáveis, reproduzíveis e sustentados por dados suficientes? |
| 5 | **Adequação** | É uma boa escolha para recuperação de documentos em português em um RAG educacional? |

> [!IMPORTANT]
> Este projeto avalia o **comportamento do modelo** em conjuntos de dados definidos. Não prova imparcialidade universal nem revela todos os dados de pré-treinamento.

---

## Estrutura do Projeto

```
minilm-evaluation/
│
├── data/
│   ├── raw/                    # Dados brutos (não modificados)
│   ├── processed/              # Dados pré-processados
│   └── test_sets/              # Datasets de avaliação
│       ├── semantic_pairs.csv  # 50 pares com notas humanas
│       ├── corpus.json         # 30 documentos sobre EJA
│       ├── queries.json        # 20 consultas de teste
│       └── relevant_docs.json  # Relevância de referência (ground truth)
│
├── notebooks/
│   ├── 01_model_inspection.ipynb      # Inspeção técnica
│   ├── 02_semantic_performance.ipynb  # Desempenho semântico
│   ├── 03_retrieval.ipynb             # Recuperação de informação
│   ├── 04_bias.ipynb                  # Avaliação de viés
│   ├── 05_robustness.ipynb            # Robustez
│   └── 06_reliability.ipynb           # Confiabilidade e comparação
│
├── results/
│   ├── metrics/    # JSONs com resultados numéricos
│   ├── plots/      # Gráficos gerados
│   └── predictions/
│
├── scripts/
│   └── run_evaluation.py   # Script CLI para execução completa
│
├── requirements.txt
└── README.md
```

---

## Instalação

### 1. Criar ambiente virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS / Codespaces
source .venv/bin/activate
```

### 2. Instalar dependências

```bash
pip install -r requirements.txt
```

### 3. Salvar versões exatas (após instalar)

```bash
pip freeze > requirements-lock.txt
```

---

## Como Executar

### Via Notebooks (recomendado — interativo)

Execute os notebooks **em ordem**:

```bash
jupyter notebook notebooks/01_model_inspection.ipynb
```

Ordem obrigatória:
1. `01_model_inspection.ipynb` — sem dependências
2. `02_semantic_performance.ipynb` — gera `semantic_pairs.csv`
3. `03_retrieval.ipynb` — gera `corpus.json`, `queries.json`, `relevant_docs.json`
4. `04_bias.ipynb` — independente
5. `05_robustness.ipynb` — independente
6. `06_reliability.ipynb` — depende de 02 e 03

### Via Script CLI (automatizado)

```bash
# Executa todas as etapas
python scripts/run_evaluation.py

# Executa só uma etapa
python scripts/run_evaluation.py --step inspection
python scripts/run_evaluation.py --step semantic
python scripts/run_evaluation.py --step retrieval
python scripts/run_evaluation.py --step bias
python scripts/run_evaluation.py --step robustness
```

> [!NOTE]
> O script CLI é mais rápido, mas os notebooks oferecem visualizações e análise exploratória mais detalhadas.

---

## Modelos Avaliados

| Modelo | ID HuggingFace | Dimensão | Multilíngue |
|--------|---------------|----------|-------------|
| **all-MiniLM-L12-v2** *(alvo)* | `sentence-transformers/all-MiniLM-L12-v2` | 384 | ✗ (principalmente EN) |
| multilingual-e5-small | `intfloat/multilingual-e5-small` | 384 | ✓ |
| paraphrase-multilingual-mpnet | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | ✓ |

> [!WARNING]
> O **multilingual-e5-small** requer prefixos: `"query: "` para consultas e `"passage: "` para documentos. Não usar prefixos resulta em desempenho inferior.

---

## Datasets de Teste

### Pares de Similaridade Semântica (`semantic_pairs.csv`)
- **50 pares** de frases em português brasileiro sobre EJA
- Notas de similaridade humana na escala 1–5
- Categorias: matrícula, conceitos, avaliação, documentação, funcionamento, linguagem informal

### Corpus de Recuperação (`corpus.json`)
- **30 documentos** sobre EJA cobrindo diferentes aspectos
- Linguagem próxima à usada em materiais educacionais reais

### Consultas de Recuperação (`queries.json` + `relevant_docs.json`)
- **20 consultas** representando perguntas reais de estudantes
- Ground truth de relevância definido manualmente

---

## Métricas de Avaliação

| Dimensão | Métrica | Interpretação |
|----------|---------|---------------|
| Similaridade | Spearman ρ | Ordenação correta dos pares |
| Similaridade | Pearson r | Correlação linear com notas humanas |
| Triplas | Acurácia | % de triplas onde positivo > negativo |
| Recuperação | Recall@K | % de docs relevantes nos top-K |
| Recuperação | Precision@K | % dos top-K que são relevantes |
| Recuperação | MRR@10 | Posição média do primeiro doc relevante |
| Recuperação | nDCG@10 | Qualidade da ordenação dos relevantes |
| Viés | Δ similaridade contrafactual | Impacto da troca de atributo identitário |
| Robustez | Sim. após negação | Risco de recuperar docs contraditórios |
| Robustez | Sim. após erro ortográfico | Tolerância a variações de escrita |

---

## Níveis de Avaliação

```
┌─────────────────────────────────────────────────┐
│  Aplicação (RAG completo)                       │
│  → RAGAS: Faithfulness, Answer Relevancy, etc.  │
├─────────────────────────────────────────────────┤
│  Sistema de Recuperação                         │
│  → Recall, Precision, MRR, nDCG                 │
├─────────────────────────────────────────────────┤
│  Modelo de Embeddings  ← este projeto           │
│  → Spearman, Pearson, triplas, viés, robustez   │
└─────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> Um bom desempenho no nível de embeddings **não garante** boa performance no RAG completo.

---

## Checklist de Qualidade

- [ ] Inspeção técnica documentada (arquitetura, parâmetros, truncamento)
- [ ] Dataset de similaridade revisado por pelo menos um avaliador humano
- [ ] Correlação Spearman e Pearson calculadas
- [ ] Avaliação de triplas executada
- [ ] Corpus, consultas e ground truth de recuperação definidos
- [ ] Métricas de recuperação (Recall, MRR, nDCG) calculadas
- [ ] Testes de viés contrafactual executados com análise de limitações
- [ ] Teste de negação executado com avaliação de risco
- [ ] Teste de robustez ortográfica executado
- [ ] Teste de mudança de domínio executado
- [ ] Versões de todas as bibliotecas registradas
- [ ] Pelo menos dois modelos alternativos comparados nas mesmas condições
- [ ] Relatório final com resultados, limitações e recomendação

---

## Referências

- [Model Card — all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)
- [Sentence Transformers — Avaliação](https://www.sbert.net/docs/package_reference/sentence_transformer/evaluation.html)
- [MTEB Benchmark](https://github.com/embeddings-benchmark/mteb)
- [Sentence-BERT (Reimers & Gurevych, 2019)](https://arxiv.org/abs/1908.10084)
- [SEAT — Sentence Encoder Association Test (May et al., 2019)](https://arxiv.org/abs/1903.03862)

---

## Citação

Se utilizar este projeto em pesquisa:

```
Avaliação experimental do all-MiniLM-L12-v2 para RAG educacional em português.
Projeto de Residência — [Instituição], 2024.
```
