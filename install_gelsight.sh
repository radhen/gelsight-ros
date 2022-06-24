#!/usr/bin/env bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )";

[ ! -d "${SCRIPT_DIR}/src/gelsightinc" ] && git clone git@github.com:gelsightinc/gsrobotics.git "${SCRIPT_DIR}/src/gelsightinc"
[ -d "${SCRIPT_DIR}/src/gelsightinc" ] && echo "GelsightSDK already in src, ignoring"

python3.8 -m pip install "${SCRIPT_DIR}/src/gelsightinc/" --upgrade

PYDIR=`python3.8 -m pip show gelsight | grep -i location | cut -f2 -d" "`

if grep -Fxq "LD_LIBRARY_PATH=${PYDIR}/gelsightcore:$LD_LIBRARY_PATH" ~/.bashrc
then
    echo "Adding gelsightcore to LD_LIBRARY_PATH"
    echo "LD_LIBRARY_PATH=${PYDIR}/gelsightcore:$LD_LIBRARY_PATH" >> ~/.bashrc
else
    echo "gelsightcore already added to LD_LIBRARY_PATH, ignoring"
fi

if grep -Fxq "PYTHONPATH=${PYDIR}/gelsightcore:${PYDIR}/gelsight:$PYTHONPATH" ~/.bashrc
then
    echo "Adding gelsightcore to PYTHONPATH"
    echo "PYTHONPATH=${PYDIR}/gelsightcore:${PYDIR}/gelsight:$PYTHONPATH" >> ~/.bashrc
else
    echo "gelsightcore already added to PYTHONPATH, ignoring"
fi



source ~/.bashrc