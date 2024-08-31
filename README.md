# Mona Lisa Override

## Setup

- `python -m venv --system-site-packages .venv` (system-site-packages so we get the `picamera` package.)
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `sudo apt-get install unclutter`


## Test

-  `python run_display.py`


## Run in Production

Start a service with *systemd*. This will start the program when the computer starts and revive it when it dies:

- `mkdir -p ~/.config/systemd/user`

- Paste the contents of `display.service` into `~/.config/systemd/user/display.service`

Start the service using the commands below:

- `systemctl --user daemon-reload`
- `systemctl --user enable display.service`
- `systemctl --user start display.service`

Start it on boot: `sudo loginctl enable-linger pi`

Get the logs: `journalctl --user -u display.service`


## Benchmark

- 20 seconds to update


## TODO

- [ ] Make frame
- [ ] Change dimensions of mona lisa to match monitor
- [ ] speed up!
    - Model quantization
    - Intel's Movidius Neural Compute Stick 2 (or equivalent)
    - Reduce Input Size
    - Efficient Preprocessing
