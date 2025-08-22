# Mona Lisa Override

## Setup

- `git clone git@github.com:camoverride/mona_lisa_override.git`
- `cd mona_lisa_override`

If using Ubuntu, do these additional commands:
- `sudo apt-get update`
- `sudo apt-get install python3-dev build-essential`

Continue:
- `python -m venv --system-site-packages .venv` (system-site-packages so we get the `picamera` package.)
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `sudo apt-get install unclutter`
- copy `inswapper_128.onnx` to the base directory of this repo.

If using Ubuntu, you need to follow an additional step to rotate the screen [link](https://askubuntu.com/questions/1266783/how-to-rotate-screen-on-wayland)
- `curl https://gitlab.com/Oschowa/gnome-randr/-/raw/master/gnome-randr.py -o gnome-randr.py`
- `chmod +x gnome-randr.py`


## Test

- `export DISPLAY=:0`
- `python run_display.py`


## Run in Production

Start a service with *systemd*. This will start the program when the computer starts and revive it when it dies. This is expected to run on a Raspberry Pi 5:

- `mkdir -p ~/.config/systemd/user`
- `cat display.service > ~/.config/systemd/user/display.service`

Start the service using the commands below:

- `systemctl --user daemon-reload`
- `systemctl --user enable display.service`
- `systemctl --user start display.service`

Start it on boot: `sudo loginctl enable-linger cam`

Get the logs: `journalctl --user -u display.service`


## Increase System Longevity

Follow these steps in order:
- Install tailscale for remote access and debugging.
- Configure backup wifi networks
- Configure a Read-Only Overlay Filesystem
- Set up periodic reboots (cron job)


## Benchmark

- ~20 seconds to update on Pi 5



TEST PHOTO to check camera visibility!
