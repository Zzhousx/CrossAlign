# CrossAlign

Requires Python 3.12 and Git.

```sh
python -m pip install -r requirements.txt
python fetch_data.py
python run_all.py
```

Regenerate figures from existing or bundled reference results:

```sh
python stages/manuscript_plots.py
python stages/plot_target_diagnostics.py
```

Quick synthetic test:

```sh
python tests/smoke.py
```
