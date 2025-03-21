🛠️ Environment Setup & Project Initialization

1. Create a new Conda environment
   

```bash
conda create -n wineq python=3.9 -y
```
2. Activate the environment
``` bash
conda activate wineq 
```
📦 Project Dependencies

3. Create a requirements file
``` bash
touch requirements.txt
``` 


4. Install dependencies from the requirements file
``` bash
 pip install -r requirements.txt

```
📥 Download Dataset

Download the dataset manually from the following link and place it in the appropriate directory:
``` bash

https://drive.google.com/drive/folders/18zqQiCJVgF7uzXgfbIJ-04zgz1ItNfF5?usp=sharing 
``` 

5. Initialize a Git repository
``` bash 
git init 
``` 

6. Initialize DVC in the project
``` bash
dvc init 
``` 

7. Track the dataset with DVC
``` bash
dvc add data_given/winequality.csv
``` 

8. Add files to Git and make the initial commit
``` bash
git add .
``` 

``` bash
git commit -m " first commit" 
```

✅ Testing & Tox

9. Run tox to test your environment setup
``` bash
tox command
```

10. Rebuild tox environments (if needed)
```bash
for rebuilding
```

🧪 Run Unit Tests

11. Run all tests using Pytest

```bash
pytest -v
```
📦 Packaging Your Project

12. Install the project as a local package


```bash
pip install -e.
```

13. Build your own Python package
```bash
python setup.py sdist bdist_wheel
```
