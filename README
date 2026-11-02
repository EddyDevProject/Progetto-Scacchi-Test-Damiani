# Timed GNN vs LLM per i puzzle "matto in n"

Progetto 2 – _Developing and Evaluating a Timed Graph Neural Network for Chess Puzzle Solving Using Lichess Data_.

**Gruppo:** Clara Palloni, Andrea Tringali, Edoardo Bavaro

Il progetto confronta una graph neural network con informazione temporale, addestrata con la libreria **TimeGNN** di Florence Wong ed Ernesto Damiani (usata senza modifiche), con un LLM (qwen3:8b via Ollama) nella guida alla soluzione di puzzle "matto in n" di Lichess, con n da 1 a 10.

Il report completo, con metodologia, grafici, test statistici e limiti, è in [`report.pdf`](report.pdf).

## Struttura della repository

```
├── progetto_timegnn_llm.ipynb     ← tutto il progetto: dati, training, valutazione, analisi
├── report.pdf                     ← report finale
├── TimeGNN-main/                  ← libreria di Wong e Damiani (non modificata)
├── results/                       ← esecuzione finale
│   ├── M1_s0.csv … M4c_s2.csv     ← esito di ogni puzzle per variante e seed
│   ├── M1_s0.pt  … M4c_s2.pt      ← modelli addestrati (pesi migliori)
│   ├── M*_history.csv             ← curve di training
│   ├── M4_confidenza.csv          ← confidenza della GNN su ogni puzzle di test
│   ├── greedy_s0.csv, random_s0.csv, LLM_qwen3_8b_s0.csv
│   └── valutazione_umana*.csv     ← voti della valutazione umana e chiave del blind
```

`data/` non è nella repository: il notebook scarica il database Lichess (circa 300 MB) e ricostruisce dati e test set da solo.

## Come eseguire

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install torch torch_geometric "pandas<3" python-chess zstandard pyarrow scipy matplotlib seaborn requests ipykernel
pip install ./TimeGNN-main
brew install stockfish ollama        # noi abbiamo addestrato su un MAC quindi per replicare usare il comando
ollama serve &                       # in un altro terminale
ollama pull qwen3:8b
```

Aprite `progetto_timegnn_llm.ipynb` con il kernel del `.venv`:

- `SMOKE = True`: verifica veloce, pochi minuti su pochi dati;
- `SMOKE = False`: esperimento completo. Sul MacBook Pro M5 (GPU MPS) servono circa 11 s per epoca di training e circa 2 ore per la valutazione dell'LLM.

Le run già completate vengono saltate, quindi il notebook si può interrompere e riprendere.

Note tecniche:

- **pandas deve essere < 3**: con pandas 3 la libreria azzera i tempi.
- La libreria **non è modificata**. Le correzioni (self-loop nel modello time-decay, tempo causale sugli archi) sono applicate ai dati. Una riga nel notebook risolve un'incompatibilità con PyTorch Geometric ≥ 2.5.

## Varianti addestrate

| Variante | Modello TimeGNN         | Tempo                                    |
| -------- | ----------------------- | ---------------------------------------- |
| M1       | `DualGATModel`          | no                                       |
| M2       | `DualGATModel`          | sì (feature del nodo + archi)            |
| M3       | `DualGATTimeAwareModel` | no                                       |
| M4       | `DualGATTimeAwareModel` | sì (decadimento dell'attenzione e^(−λt)) |
| M4c      | `DualGATTimeAwareModel` | tempo costante (controllo)               |

Tre seed per variante, 45.616 puzzle di training (n = 1–5), 1.790 puzzle di test (n = 1–10).
