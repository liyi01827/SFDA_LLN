#!/bin/bash

# Train with SHOT method
cd ./sfda_lln/shot
python train_visda.py       ~/data/visda/  
python train_officehome.py  ~/data/oh         --all True
python train_office31.py    ~/data/o31        --all True
python train_domainnet.py   ~/data/domainnet  --all True


# Train with NRC method
cd ./sfda_lln/nrc
python train_visda.py           
python ./office_home/train_tar.py             --all True
python ./offic31/train_tar.py                 --all True
python train_domainnet.py   ~/data/domainnet  --all True


# Train with G-SFDA method
cd ./sfda_lln/gsfda
python train_tar_visda.py  
python train_tar_oh.py                  --all True
python train_tar_o31.py                 --all True
python train_tar_domain.py    ~/data/domainnet  --all True
