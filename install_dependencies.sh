sudo apt-get update
# require python 3.8
echo "Installing Python 3.8"
sudo apt-get install python3.8

# additional dependencies, mostly for pygame, asyncio and pandas
sudo apt-get install python-dev libsdl-ttf2.0-dev libsmpeg-dev python-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev

# sdl-config
sudo apt install libsdl2-dev libsdl2-2.0-0 -y;

# sdl-image
sudo apt install libwebp-dev libtiff5-dev libsdl2-image-dev libsdl2-image-2.0-0 -y;

#install sdl mixer
sudo apt install libmikmod-dev libfishsound1-dev liboggz2-dev libflac-dev libfluidsynth-dev libsdl2-mixer-dev libsdl2-mixer-2.0-0 -y;


echo "Installing packages and libraries from requirements.txt"
python3.8 -m pip install -r requirements.txt


