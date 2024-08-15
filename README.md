# Mona Lisa Override

## Setup

- `python -m venv --system-site-packages .venv` (system-site-packages so we get the `picamera` package.)
- `source .venv/bin/activate`
- `pip install -r requirements.txt`


## Run

-  `python run_display.py`


## Benchmark

- 20 secs to reach `print("Identifying faces")`, 20 secs more to finish -- not bad!


## TODO

- change dimensions of mona lisa to match monitor
- get to run automatically
- speed up!
