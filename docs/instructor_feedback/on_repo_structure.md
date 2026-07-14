Please use the following general guidelines, structure and placeholders for creating your github repository. You may modify this to suit your specific needs. This will evolve over a period of your project duration. I will share with you specific requirements when you get to the deployment phase.

project-name/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt (or environment.yml / pyproject.toml)
├── CHANGELOG.md
│
├── docs/
│   ├── Project_Proposal.pdf
│   ├── Milestone1_Report.pdf
│   ├── Milestone2_Report.pdf
│   ├── Final_Report.pdf
│   ├── Architecture_Diagram.png
│   └── Presentation.pdf
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── sample/
│   └── README.md
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Model_Experiments.ipynb
│   └── 04_Evaluation.ipynb
│
├── src/
│   ├── data/
│   ├── models/
│   ├── inference/
│   ├── retrieval/
│   ├── agents/
│   ├── evaluation/
│   ├── utils/
│   └── config.py
│
├── app/
│   ├── app.py
│   ├── pages/
│   ├── static/
│   └── templates/
│
├── configs/
│   ├── model.yaml
│   ├── train.yaml
│   └── inference.yaml
│
├── models/
│   ├── checkpoints/
│   └── README.md
│
├── outputs/
│   ├── predictions/
│   ├── reports/
│   ├── figures/
│   └── logs/
│
├── tests/
    ├── test_data.py
    ├── test_model.py
    └── test_pipeline.py

**README.md**
The README should contain:
- Project overview
- Problem statement
- Features
- Architecture diagram
- Dataset details
- Model(s) used
- Installation steps
- Running instructions
- Results
- Demo screenshots
- Team members
- References

**docs/**
This folder is is used to store to provide a complete project history in one place.
- Milestone reports
- Final report
- Architecture diagrams
- Presentations
- User manual

**data/**
Do not commit large datasets. Instead:
- Keep only sample data.
- Include scripts or instructions for downloading datasets.
- Add a README describing the dataset sources and preprocessing.

**notebooks/**
Use notebooks for:
- Exploratory Data Analysis
- Initial experiments
- Visualizations
Once codes stabilize, move it into src/.

**src/**
This should contain the reusable project code.
For example,:
src/
    models/
        yolo_detector.py

    retrieval/
        rag_pipeline.py

    inference/
        generate_report.py

    evaluation/
        metrics.py

    utils/

**app/**
Contains the deployment interface (Streamlit, Gradio, or Flask), separate from the core AI logic.

**outputs/**
Keep generated outputs here and not with source code.
- prediction files
- evaluation reports
- confusion matrices
- plots
- screenshots

**tests/**
The folder should contain the test code. The resulting output can be kept in the \output folder.

**CONTRIBUTING.md**
Defines:
- Coding conventions,
- Branch naming,
- Pull request process,
- Team responsibilities.
