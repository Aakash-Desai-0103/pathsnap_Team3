# PathSnap: Map-Matching & Route Reconstruction
UE24CS352A Machine Learning Hackathon, PES University (Team 3)

PathSnap matches noisy GPS observations to a road network. For every GPS point it generates
nearby candidate road arcs, scores each candidate with a trained classifier, and uses an
HMM + Viterbi decoder to reconstruct a connected route. The system runs as a FastAPI backend
(Laptop 1) and a browser UI (Laptop 2).

## Repository structure
| Folder | Contents |
|---|---|
| `notebooks/` | `01_eda.ipynb` (EDA, preprocessing, labels, candidates, features), `02_model_experiments.ipynb` (model comparison, Bayesian optimisation), `03_hmm_viterbi.ipynb` (HMM vs classifier only) |
| `src/` | `pipeline.py` (prediction pipeline: preprocessing, candidates, features; never reads `.route`), `hmm.py` (HMM + Viterbi decoder), `scoring.py` (server-side scoring, the only code that reads `.route`) |
| `api/` | `main.py` (FastAPI: `/health`, `/trips`, `/predict/{trip_id}`), `config.json` |
| `ui/` | `index.html` (evaluator interface, Leaflet map) |
| `models/` | `final_model.joblib` (final classifier: MLP with 15 features) |
| `results/` | experiment logs, metrics tables, plots |
| `maps/` | Folium maps |
| `Manuals/` | hackathon guideline documents |
| `data/` | raw trips (not committed; see below) |

## Data (not committed)
Place the trip folders locally:
```
data/student_test/<trip_id>/<trip_id>.track|.nodes|.arcs|.route
data/evaluator_test/<trip_id>/...
```
`data/` is gitignored. Ground-truth `.route` files are read only by `src/scoring.py` on the server, after prediction.

## Setup
```
pip install -r requirements.txt
```
Use the same scikit-learn version as the one used to train the model (Google Colab).

## Running the demo
**Laptop 1 (API)**, from the repo root:
```
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- `api/config.json` selects the data folder (`data/evaluator_test` or `data/student_test`), the model, and `use_hmm`.
- Check `http://localhost:8000/health`. Find the laptop's IP with `ipconfig`.
- Allow Python through Windows Firewall (Private and Public) so Laptop 2 can connect.

**Laptop 2 (UI)**, same Wi-Fi or hotspot:
```
cd ui
python -m http.server 5500
```
Open `http://localhost:5500`, set the API address to `http://<Laptop-1-IP>:8000`, click **Connect & load trips**, then select a trip.
Every request is logged in the Laptop 1 terminal.

## Method summary
- **Preprocessing:** per-trip local metric projection, 60 s gap segmentation, heading and speed.
- **Labels:** monotone dynamic-programming alignment of GPS points to the ordered reference route; points > 50 m from the route are flagged and excluded.
- **Candidates:** KD-tree over arcs densified every 10 m; R = 50 m, K = 10 (candidate miss rate 0.66%).
- **Features:** 9 base features (distance, rank, heading difference, projection position, ...) + 6 context features (distance to neighbouring GPS points, 6 s heading).
- **Models:** 7 algorithm families compared with GroupKFold-by-trip CV; final model MLP (15 features).
- **Bayesian optimisation:** Optuna (TPE + median pruning), 20 trials.
- **HMM + Viterbi:** states = candidate arcs; emissions = classifier probabilities; transitions = road connectivity and distance consistency, with U-turn and jump penalties.

## Results
| Stage | Top-1 accuracy |
|---|---|
| Nearest arc (no ML), student_test | 54.8% |
| Best classifier, 9 features, student_test | 85.1% |
| MLP + context features, student_test | 89.2% |
| MLP + context + HMM, student_test | 92.6% |
| **Final system on the 10 evaluator trips** | **94.5%** |

Detailed tables are in `results/`.

## AI Assistance Disclosure
AI tools (Claude) were used for guidance and code assistance. All experiments were run and analysed by the team.
