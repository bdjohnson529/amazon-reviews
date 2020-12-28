:: virtual environment
:: for some reason installation does not work on some machines
call conda remove -y -n amazonreviews --all
call conda env create -f environment.yml
call conda activate amazonreviews