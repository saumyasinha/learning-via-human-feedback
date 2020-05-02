# TAMER-ER: Augmenting TAMER with Facial Expression Recognition

<img src=https://github.com/saumyasinha/learning-via-human-feedback/blob/master/assets/mountain_car.png width="800" height="400" align="middle">
### Installation and Dependencies
Requires Python 3.8.1 or higher.

Clone the git repository
```
git clone https://github.com/saumyasinha/learning-via-human-feedback
cd learning-via-human-feedback
git pull
git lfs pull
```

#### Linux
Install packages and libraries using `sh install_dependencies.sh`. This will install Python 3.8 and install all the necessary modules required to run TAMER-ER.

#### Mac OSX

Install Python 3.8.1 or higher and verify version using `python3 -V`

Optionally, we recommend using virtual environment (`venv`) to run our code. Here is an example:

```
python3 -m venv new_env
source new_env/bin/activate
```

Install the packages and libraries by running the following lines:

```
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```


### Running TAMER-ER

Use `python3 run_tamerer.py -o demo` to run our code and direct output to directory `demo`.

Use `python3 run_tamerer.py --help` to show usage information.


## Authors and Citation

This repo is the implementation of **Augmenting TAMER with Facial Expression Recognition: TAMER-ER**. If you use our work, please cite us. You can find [our paper here](https://github.com/saumyasinha/learning-via-human-feedback/blob/master/assets/AFHRI_Final_Paper_Draft.pdf)

[Beni Bienz](https://github.com/benibienz), University of Colorado Boulder.

[Christine Chang](http://www.xtinebot.com), University of Colorado Boulder.

[Michael Lauria](https://github.com/mikedeltalima), University of Colorado Boulder.

[Vikas Nataraja](https://github.com/vikasnataraja), University of Colorado Boulder.

[Saumya Sinha](https://github.com/saumyasinha), University of Colorado Boulder.

Special thanks to Dr. Bradley Hayes for his assistance and guidance.
