create env 
```bash
conda create -n wineq python=3.9 -y

```

activate env
``` bash
conda activate wineq 

```
create a req file 
``` bash
touch requirements.txt
``` 


insatll requirements
``` bash
 pip install -r requirements.txt

```
``` bash
download the data from this link 

https://drive.google.com/drive/folders/18zqQiCJVgF7uzXgfbIJ-04zgz1ItNfF5?usp=sharing 
``` 

``` bash 
git init 
``` 

``` bash
dvc init 
``` 

``` bash
dvc add data_given/winequality.csv
``` 

``` bash
git add .
``` 

``` bash
git commit -m " first commit"

``` 
tox command
``` bash
tox
```
for rebuilding
```bash
tox -r
```
pytest command
```bash
pytest -v
```

setup commands -
```bash
pip install -e.
```

build your own package commands-
```bash
python setup.py sdist bdist_wheel
```