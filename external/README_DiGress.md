python main.py \
  dataset=sbm \
  +experiment=sbm.yaml \
  dataset.datadir=/path/to/x \
  general.name=test \
  general.wandb=disabled \
  general.gpus=1 \
  train.seed=0 \
  train.n_epochs=1000 \
  train.batch_size=12 \
  train.save_model=True \
  general.check_val_every_n_epochs=20 \
  general.sample_every_val=100000000 \
  hydra.run.dir=/path/to/runs/digress_x_seed0