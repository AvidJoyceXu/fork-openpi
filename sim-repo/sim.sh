# Run the simulation
XLA_FLAGS="--xla_gpu_autotune_level=0" \
PYTHONPATH=/home/lingyun/openpi/.venv/lib/python3.11/site-packages \
MUJOCO_GL=egl \
python examples/aloha_sim/main.py