# ctrl
Redwood-2



## Setup
Make sure you have Python installed on your machine.
```bash
python -m venv venv
pip install -r requirements.txt
pip install -e .
```

Next we need to install APPS dataset. Run the following command to download the dataset.

```bash
mkdir data
curl https://people.eecs.berkeley.edu/~hendrycks/APPS.tar.gz -o data/APPS.tar.gz
tar -xvf data/APPS.tar.gz -C data/
```

### Test different protocols


#### Trusted Monitoring
```bash
python -m src.experiments.five_shot
```

#### Task Decomposition
```bash
python -m src.experiments.task_decomposition
```