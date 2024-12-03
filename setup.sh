# create an venv and install dependencies
. /etc/profile.d/modules.sh

# Check if the virtual environment already exists
# the cluster default is Python 3.12.3
if [ ! -d ".env" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .env
    source .env/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "Virtual environment already exists. Activating..."
    source .env/bin/activate
fi

#setup modules
echo "Loading modules..."

module add gcc/11
module add cuda/12.1

# show currently loaded modules
module list