mkdir checkpoints > /dev/null 2>&1
mkdir logs > /dev/null 2>&1
python3 ga3c_gazebo/GA3C.py "$@"
