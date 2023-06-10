python train_pdm.py --dataset cora --device cuda --depth 1 --cls-lr 1e-2 --cls-min-lr 1e-3

python train_pdm.py --dataset citeseer --device cuda --depth 1 --lr 1e-3 --min-lr 1e-4

python train_pdm.py --dataset pubmed --device cuda --depth 2 --batch-size 2048 --rep-size 32 --activation gelu --last-activation tanh --diffusion ppr --alpha 0.06 --cls-min-lr 1e-3