# Installation

## Instal pythermalfunctions

**You can skip this part if you prefer to use the J-functions
implemented in CosmoTransitions.**

Install `pythermalfunctions` for the thermal J-functions.

First install `fpm` if not already installed:
```bash
wget https://github.com/fortran-lang/fpm/releases/download/v0.13.0/fpm-0.13.0-linux-x86_64-gcc-12
mv fpm-0.13.0-linux-x86_64-gcc-12 ~/.local/bin/fpm
chmod +x ~/.local/bin/fpm
```

Then install `patchelf` if not already installed:
```bash
pip install patchelf
```

Now install the python wrapper for `pythermalfunctions`:
```bash
git clone https://github.com/thomasbiekoetter/thermalfunctions
cd thermalfunctions
fpm build --flag="-fPIC"
cd python
make wrapper
```

Check if everything works (requires cosmoTransitions):
```bash
pip install cosmoTransitions
cd test
python compare_cosmoTransitions.py
```

## Install this package

From the root directory of this repo run:
```bash
pip install -e .
```
