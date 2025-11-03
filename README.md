# Legal-Analyst

A collection of Jupyter notebooks and Python utilities for analyzing legal documents, contracts, and court filings. This repository provides tools for parsing PDFs and Word files, extracting entities and citations, producing human-readable summaries, and performing basic classification and risk scoring — ideal for researchers, lawyers, and legal technologists prototyping NLP workflows.

Table of contents
- Overview
- Key capabilities
- Notebooks and what they do
- Installation
- Quick start
- Example workflows
- Data sources and privacy
- Models and evaluation
- Project structure
- Development
- Contributing
- License
- Contact

Overview
Legal-Analyst bundles exploratory notebooks and reusable Python code to accelerate legal text analysis. It demonstrates common tasks:
- Document ingestion (PDF/Word/HTML)
- Text cleaning and canonicalization
- Named entity recognition (parties, dates, monetary amounts, statutes)
- Citation extraction and normalization
- Summarization (extractive & abstractive)
- Clause classification and risk highlighting
- Basic similarity search and clustering

Key capabilities
- End-to-end notebooks showing ingestion -> preprocess -> analysis -> visualization.
- Tokenization and NER pipelines (spaCy / transformers).
- Off-the-shelf summarization examples (transformers / OpenAI).
- Heuristics and regex modules for legal citation extraction.
- Tools for evaluating classifiers and summarizers on labeled examples.

Notebooks and what they do
- notebooks/00_setup_and_data_ingest.ipynb
  - Environment setup and how to load PDFs, DOCX, and text files.
- notebooks/01_preprocessing_and_cleaning.ipynb
  - Text normalization, sentence segmentation, de-noising.
- notebooks/02_named_entity_extraction.ipynb
  - Use spaCy and transformer-based NER to extract people, organizations, dates, monetary values.
- notebooks/03_citation_extraction_and_normalization.ipynb
  - Identify case law citations, statutes, and normalize them for downstream use.
- notebooks/04_summarization_and_abstraction.ipynb
  - Compare extractive and abstractive summarization approaches and show examples.
- notebooks/05_clause_classification_and_risk_scoring.ipynb
  - Train simple classifiers to label common contract clauses and assign risk scores.
- notebooks/06_evaluation_and_metrics.ipynb
  - Precision/recall/F1 for extractors and ROUGE/BLEU for summarizers.

Installation
- Python: 3.8+
- Recommended packages:
  - jupyterlab, pandas, numpy, scikit-learn
  - spaCy, transformers, torch
  - pdfminer.six, python-docx, textract (optional)
  - sentence-transformers (for semantic search)
- Install:
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt

Quick start
1. Start Jupyter:
   jupyter lab
2. Open notebooks/00_setup_and_data_ingest.ipynb and follow the instructions to point the ingestion cells at a /data folder with sample documents.
3. Run the next notebooks in order to explore workflows and experiments.

Example workflows
- Contract review:
  1. Ingest a contract PDF.
  2. Run preprocessing to split sections and normalize headings.
  3. Run clause classification to identify indemnity, termination, confidentiality clauses.
  4. Produce a short summary and a risk report highlighting high-risk clauses.
- Case law summarization:
  1. Ingest court opinions.
  2. Extract citations and legal holdings.
  3. Produce abstracts and tag precedential value.

Data sources and privacy
- This repository includes demo data only. Do NOT commit or include sensitive client data in the repository.
- For working with real legal data, follow your organization's privacy and retention policies and consider on-prem or private-cloud computation for confidentiality.

Models and evaluation
- The notebooks show how to train and evaluate options:
  - spaCy for NER when you have modest labeled data.
  - Transformer-based models (fine-tuned) for higher-quality NER and summarization.
  - Use ROUGE or BERTScore for summarization evaluation.
- There's a sample evaluation notebook (notebooks/06_evaluation_and_metrics.ipynb) that demonstrates metrics and cross-validation.

Project structure (suggested)
- notebooks/                # Exploratory analysis and tutorials
- src/
  - legal_analyst/
    - ingest.py
    - preprocess.py
    - ner.py
    - citations.py
    - summarizer.py
    - classifiers.py
- data/
  - sample/                 # Small, non-sensitive sample documents for tutorials
- models/
- tests/
- requirements.txt
- README.md

Extending the project
- Add new dataset loaders into src/ingest.py for different file types or APIs.
- Plug in new transformer models by extending src/summarizer.py.
- Add a REST wrapper (FastAPI/Flask) if you want to serve analysis as a service.

Contributing
- Issues and PRs welcome. For larger changes, open an issue describing the proposed work first.
- When adding models or large datasets, keep models and data out of repository LFS by default (or add to /models with a download script).

Ethics and legal considerations
- Be cautious when automating legal advice. This project is for analysis and prototyping — not a replacement for licensed legal counsel.
- Clearly label any outputs as automated and include disclaimers when used for decision-making.

License
- Add a license file appropriate to your intended use. MIT is a common permissive choice; for commercial or academic, choose accordingly.

Contact
- Maintainer: HeroicKrishna160905
- For help, open issues on this repository.

Notes
This README is structured to be adaptable: once you confirm the exact notebooks and modules present in the repo, I can tailor the descriptions and example commands to match files, function names, and any required credentials or dependencies.
