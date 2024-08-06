# DeepThinkingGNN
GNN for node/graph classification via logical extrapolation.

## Requirements

- Python 3.8 or later
- PyTorch 2.2.0 
- Torch Geometric 2.0.8 
- Torchvision 0.11.1
- NumPy 1.21.2
- Scikit-learn 1.5.0

## Installation

Changing "*prefix: /path/to/your/env/deept*" in .yaml file:
```bash
conda env create -f environment.yaml
conda activate deept
```

Otherwise cloninig the repository:

```bash
git clone https://github.com/LorenzoBini4/DeepThinkingGNN.git
cd DeepThinkingGNN
```
Via requirements
```bash
pip install -r requirements.txt
````

## Run
```bash
python3 -u main.py --num_projection_layers 1 --num_recurrent_layers 4 --num_output_layers 1 --train_iterations 20 --test_iterations 80
```
