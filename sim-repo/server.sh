# Run the server
XLA_FLAGS="--xla_gpu_autotune_level=0" \
PYTHONPATH=/home/lingyun/openpi/.venv/lib/python3.11/site-packages \
uv run scripts/serve_policy.py --env ALOHA_SIM