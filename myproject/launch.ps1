cd c:/Users/Mayn/Desktop/myfiles/PINN/myproject/
conda activate pinn-seagrass
python train_latent_physics_pinn.py   --epochs 3000   --synthetic-data pinn_training_data_synth.mat   --synthetic-force-weight 0.35   --synthetic-aux-weight 0.50
pause