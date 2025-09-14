# Mona Lisa Override 🧑‍🎨

Code for my face-swapping Mona Lisa portrait.

Battle-tested and ready for long run times in gallery settings!

![View Mona Lisa Override](images/mona_lisa_override.png)


## Setup

Ubuntu:

- `git clone git@github.com:camoverride/mona_lisa_override.git`
- `cd mona_lisa_override`
- `python -m venv .venv`
- `source .venv/bin/activate`
- `sudo apt-get update`
- `sudo apt-get install python3-dev build-essential`
- `pip install -r requirements.txt`
- `sudo apt-get install unclutter`
- `curl https://gitlab.com/Oschowa/gnome-randr/-/raw/master/gnome-randr.py -o gnome-randr.py`
- `chmod +x gnome-randr.py`
- copy `inswapper_128.onnx` to the base directory of this repo.

Raspberry Pi w/ Picam:

- `git clone git@github.com:camoverride/mona_lisa_override.git`
- `cd mona_lisa_override`
- `python -m venv --system-site-packages .venv` (system-site-packages so we get the `picamera` package.)
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `sudo apt-get install unclutter`
- `curl https://gitlab.com/Oschowa/gnome-randr/-/raw/master/gnome-randr.py -o gnome-randr.py`
- `chmod +x gnome-randr.py`
- copy `inswapper_128.onnx` to the base directory of this repo.

MacOS (for testing):

- `git clone git@github.com:camoverride/mona_lisa_override.git`
- `cd mona_lisa_override`
- `python -m venv .venv`
- `source .venv/bin/activate`
- `pip install -r requirements.txt`
- `curl https://gitlab.com/Oschowa/gnome-randr/-/raw/master/gnome-randr.py -o gnome-randr.py`
- `chmod +x gnome-randr.py`
- copy `inswapper_128.onnx` to the base directory of this repo.


## Test

Test the camera using `cheese` to make sure the scene is visible.

Run the code:

- `python run_display.py`


## Run in Production

Start a service with *systemd*. This will start the program when the computer starts and revive it when it dies. This is expected to run on a Raspberry Pi 5 or Beelink running Ubuntu:

- `mkdir -p ~/.config/systemd/user`
- `cat display.service > ~/.config/systemd/user/display.service`

Start the service using the commands below:

- `systemctl --user daemon-reload`
- `systemctl --user enable display.service`
- `systemctl --user start display.service`

Start it on boot: 

- `sudo loginctl enable-linger $(whoami)`

Get the logs: 

- `journalctl --user -u display.service`


## Increase System Longevity

Follow these steps in order:

- (1) Install tailscale for remote access and debugging.
- (2) Configure backup wifi networks.
- (3) Configure a Read-Only Overlay Filesystem
- (4) Set up periodic reboots (systemd).


## Benchmark

- ~20 seconds to update on Pi 5
- ~1 sec on Beelink (Ubuntu)
