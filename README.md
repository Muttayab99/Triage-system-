# Support Triage Pipeline

A replayable support-triage pipeline that reads tickets and config from disk, runs deterministic normalization, performs LLM triage, supports human review overrides, and produces final queue outputs.

## Project Layout

- input/ - input files
  - tickets.json
  - triage_config.json
- output/ - generated artifacts
- src/ - core pipeline modules
- main.py - CLI pipeline entrypoint
- app.py - Streamlit UI

## Setup

```powershell
python -m pip install -r requirements.txt
```

Copy the env template and set your key:

```powershell
copy .env.example .env
```

## Run the Pipeline

```powershell
python main.py
```

Outputs are written to output/ by default.

## Validate Outputs

```powershell
python validate.py
```

## Streamlit App

```powershell
streamlit run app.py
```

## Notes

- Inputs are always read from input/tickets.json and input/triage_config.json unless overridden by CLI args.
- The only interactive step is the human review overrides in the CLI; the Streamlit UI provides a review table.
- Generated artifacts and secrets are excluded by .gitignore.
