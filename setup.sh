# create an venv and install dependencies
# the cluster default is Python 3.12.3
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt

#setup modules
. /etc/profile.d/modules.sh
module add gcc/11
module add cuda/12.1

# show currently loaded modules
module list