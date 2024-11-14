# Mona Lisa Override

## Setup

- `git checkout mona_list_chicago`
- `python -m venv --system-site-packages .venv` (system-site-packages so we get the `picamera` package.)
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `sudo apt-get install unclutter`


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


## Benchmark

- 20 seconds to update


## TODO

- [ ] Configure a Read-Only Overlay Filesystem
- [ ] Remove print statements
- [ ] Store Logs in RAM (Optional): For system logs or any logs your code might generate, consider redirecting /var/log
- [ ] Periodic Reboots (cron job)
- [ ] Limit Background Services and Packages: Install only the essential software needed for your project, and disable unnecessary background services to reduce resource use and potential points of failure.
- [ ] Minimize OS-Level Writes: Reduce logging and disable unnecessary services that may write to the SD card, or set up /tmp and other temporary directories to store data in RAM instead of on the SD card.
