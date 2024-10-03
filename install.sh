### exit on error
set -e
### set conda hook
eval "$(conda.shell.bash hook)"

### create enviroment
conda create -n RAM -y python=3.7
conda activate RAM

### install package
pip install -r requirements.txt
python setup.py develop
