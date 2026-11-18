# Adversarial Attacks against Deep Learning-Based Email Phishing Detection Systems

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
![Python](https://img.shields.io/badge/python-3.x-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Author:** Alabi Ayobami James, Machine Learning Engineer and Researcher
**Affiliation:** LibertyEdge Tech Solutions (Research Team). Published with the organisation's permission.
**Full report (PDF):** [Google Drive](GOOGLE_DRIVE_LINK) · [`docs/report.pdf`](docs/report.pdf)
**Report date:** September 2025

---

## Summary

Deep learning models can detect phishing emails with high accuracy, but small text manipulations (for example character swaps) can make them misclassify phishing emails as safe. This project:

1. builds an LSTM classifier for phishing email detection,
2. attacks it with the TextAttack **DeepWordBug (Gao et al., 2018)** recipe,
3. retrains it with adversarial examples added to the training data (adversarial training), and
4. compares the original and retrained models on clean accuracy and attack resistance.

The baseline model reached **97.8 % test accuracy** on clean data. After adversarial retraining, it reached **96.82 %** and showed a lower attack success rate on the attacked sample (30 % → 20 %, see [Limitations](#limitations-and-honest-caveats) for how much weight this can bear).

## My contributions

All implementation, experiments and analysis in this repository were carried out by the author:

- Exploratory data analysis of the phishing email dataset
- Text cleaning, tokenisation and TF-IDF feature engineering
- Design, training and evaluation of the LSTM classifier
- Implementation of the adversarial attack pipeline (TextAttack, DeepWordBugGao2018)
- Adversarial training: building the augmented dataset and retraining the model
- Evaluation and comparison (accuracy, precision, recall, F1, ROC-AUC, confusion matrices, attack success rate, query counts)
- Writing the full research report

## Repository structure

```
.
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_processing_model_development.ipynb
│   ├── 03_adversarial_attack_initial_model.ipynb
│   ├── 04_retraining_model_with_adversarial_examples.ipynb
│   └── 05_comparison_initial_vs_retrained.ipynb
├── src/
│   ├── attack_model.py
│   ├── retrain_model.py
│   ├── evaluate_robust_model.py
│   └── compare_models_visualizations.py
├── data/            # see data/README.md (dataset is not redistributed)
├── models/          # trained models (see models/README.md)
├── results/figures/ # plots used in the report
├── docs/report.pdf  # full research report
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

Run the notebooks **in numerical order**; each one builds on the outputs of the previous.

## Dataset

Kaggle *Phishing Email Detection* dataset (`subhajournal/phishingemails`), with an email text column and an email type column (*Phishing Email* / *Safe Email*). The data is not included in this repository; follow [`data/README.md`](data/README.md) to download it.

## Method

| Stage | Details |
|---|---|
| Preprocessing | Removal of hyperlinks, punctuation and extra whitespace; tokenisation; integer encoding; sequence-length analysis |
| Features | TF-IDF transformation and token sequences (see notebook 02) |
| Model | LSTM binary classifier, sigmoid output, binary cross-entropy loss, Adam optimiser (learning rate 0.001) |
| Attack | TextAttack `DeepWordBugGao2018` (character-level perturbations: swaps, substitutions, deletions, insertions), black-box |
| Defence | Adversarial training: successful adversarial examples are added to the training set and the model is retrained |
| Metrics | Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, attack success rate, mean queries per attack, mean perturbed-word percentage |

## Results

### Clean test performance

| Metric | Initial model | Retrained model |
|---|---|---|
| Accuracy | 0.978 | 0.9682 |
| Phishing class precision / recall / F1 | 0.98 / 0.98 / 0.98 | 0.97 / 0.97 / 0.97 |
| ROC-AUC | 0.9964 | 0.9950 |
| False positives (safe flagged as phishing) | 33 | 59 |
| False negatives (phishing missed) | 45 | 49 |

Adversarial training cost about one percentage point of clean accuracy.

### Adversarial attack (DeepWordBugGao2018)

| Experiment | Successful | Failed | Skipped | Reported success rate |
|---|---|---|---|---|
| Attack on initial model, notebook 03 (18 emails) | 3 | 7 | 8 | 16.67 % |
| Initial model, comparison run, notebook 05 (10 emails) | 3 | 5 | 2 | 30 % |
| Retrained model, comparison run, notebook 05 (10 emails) | 2 | 8 | 0 | 20 % |

Other figures from the notebook 03 attack: accuracy under attack 70 %, average perturbed words 18.53 %, average 717.8 queries per input. In the comparison run, successful attacks needed more queries against the retrained model than against the initial one.

Figures (confusion matrices, ROC curves, attack-outcome charts) are in [`results/figures/`](results/figures/) and in Chapter 4 of the report.

## Limitations and honest caveats

- **Small attack samples.** The attacks were run on 10 to 18 emails. A change from 3/10 to 2/10 successful attacks is **not statistically significant**, so the robustness gain should be read as indicative, not conclusive. A natural extension is to repeat the attack on hundreds of test emails and report confidence intervals.
- **Two success-rate figures.** The 16.67 % figure (notebook 03) and the 30 % figure for the same model (notebook 05) come from runs of different size with different skip counts, so they are not directly comparable.
- **One attack family, one dataset, one architecture.** Only DeepWordBug was used, against an LSTM, on a single Kaggle dataset. Results may not generalise to other attacks (for example TextFooler, BERT-Attack), transformer models or other corpora.
- **Clean-performance trade-off.** Retraining slightly reduced clean accuracy and increased false positives.
- **Computational cost.** Iterative attack generation is expensive (about 700 queries per attacked email).

## Reproducing the results

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR-USERNAME/phishing-adversarial-robustness.git
   cd phishing-adversarial-robustness
   ```
2. Create an environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows (use: source .venv/bin/activate on macOS/Linux)
   pip install -r requirements.txt
   ```
3. Download the dataset as described in [`data/README.md`](data/README.md).
4. Run the notebooks in `notebooks/` from 01 to 05. The scripts in `src/` contain the attack, retraining, evaluation and plotting code as standalone modules.
5. Trained models, if included, are in `models/`.

Results can vary slightly between runs because of random initialisation and the stochastic parts of the attack.

## Citation

If you use this work, please cite it (GitHub's "Cite this repository" button uses `CITATION.cff`):

```
Alabi, A. J. (2025). Adversarial Attacks against Deep Learning-Based Email Phishing
Detection Systems. LibertyEdge Tech Solutions. https://doi.org/10.5281/zenodo.XXXXXXX
```

## Licence and acknowledgements

Code is released under the MIT licence (see [`LICENSE`](LICENSE)). This work was carried out at LibertyEdge Tech Solutions and is published with the organisation's permission. The dataset belongs to its original authors on Kaggle and is subject to its own licence. Adversarial attacks were generated with the [TextAttack](https://github.com/QData/TextAttack) library.

## Contact

Alabi Ayobami James · EMAIL_ADDRESS · [ORCID](https://orcid.org/XXXX-XXXX-XXXX-XXXX) · [LinkedIn](LINKEDIN_URL)
