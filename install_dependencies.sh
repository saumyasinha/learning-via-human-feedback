sudo apt-get update
# require python 3.8
echo "Installing Python 3.8"
sudo apt-get install python3.8

# additional dependencies, mostly for pygame, asyncio and pandas
sudo apt-get install python-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev

echo "Installing packages and libraries from requirements.txt"
python3.8 -m pip install -r requirements.txt


