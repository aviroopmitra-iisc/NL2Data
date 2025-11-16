# NL2Data Streamlit UI

A user-friendly web interface for the NL2Data pipeline.

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

This will install Streamlit and the nl2data package in editable mode.

## Running the UI

```bash
streamlit run app.py
```

Then navigate to `http://localhost:8501` in your browser.

## Usage

1. Enter a natural language description of the dataset you want to generate
2. Click "Run pipeline"
3. Watch the progress as each step completes
4. Download the generated CSV files

## Features

- Real-time step progress tracking
- Schema visualization
- CSV file downloads
- Error handling and display

