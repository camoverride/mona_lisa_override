# Mona Lisa Override

## Setup

- `git checkout mona_list_chicago`
- `python -m venv --system-site-packages .venv` (system-site-packages so we get the `picamera` package.)
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `sudo apt-get install unclutter`
- copy `inswapper_128.onnx` to the base directory of the repo.

## Test

-  `python run_display.py`


## Run in Production

Start a service with *systemd*. This will start the program when the computer starts and revive it when it dies:

- `mkdir -p ~/.config/systemd/user`
- `cat display.service > ~/.config/systemd/user/display.service`

Start the service using the commands below:

- `systemctl --user daemon-reload`
- `systemctl --user enable display.service`
- `systemctl --user start display.service`

Start it on boot: `sudo loginctl enable-linger pi`

Get the logs: `journalctl --user -u display.service`


## Increase Longevity

Follow these steps in order:
- Install tailscale for remote access and debugging.
- Configure backup wifi networks (hotspot) in wpa_supplicant.conf
- Set up periodic reboots (cron job)
- Configure a Read-Only Overlay Filesystem


## Benchmark

- 20 seconds to update
