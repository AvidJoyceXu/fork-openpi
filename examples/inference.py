# 
import dataclasses

import jax

from openpi.models import model as _model
from openpi.policies import droid_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

#  [markdown]
# # Policy inference
# 
# The following example shows how to create a policy from a checkpoint and run inference on a dummy example.

# 
config = _config.get_config("pi0_fast_droid")
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_droid")

# Create a trained policy.
policy = _policy_config.create_trained_policy(config, checkpoint_dir)
print("finished creating policy")
# Run inference on a dummy example. This example corresponds to observations produced by the DROID runtime.
example = droid_policy.make_droid_example()
'''
dict_keys([
'observation/exterior_image_1_left', 
'observation/wrist_image_left', 
'observation/joint_position', 
'observation/gripper_position', 
'prompt'
])
'''
print("image shape:")
print(example["observation/exterior_image_1_left"].shape)
print(example["observation/wrist_image_left"].shape)
print("joint position shape:")
print(example["observation/joint_position"].shape)
print("gripper position shape:")
print(example["observation/gripper_position"].shape)
print("prompt:")
print(example["prompt"])

result = policy.infer(example)
print("finished inferring")

# Delete the policy to free up memory.
del policy

print("Actions shape:", result["actions"].shape)

#  [markdown]
# # Working with a live model
# 
# 
# The following example shows how to create a live model from a checkpoint and compute training loss. First, we are going to demonstrate how to do it with fake data.
# 

# 
config = _config.get_config("pi0_aloha_sim")

checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_aloha_sim")
key = jax.random.key(0)

# Create a model from the checkpoint.
model = config.model.load(_model.restore_params(checkpoint_dir / "params"))

# We can create fake observations and actions to test the model.
obs, act = config.model.fake_obs(), config.model.fake_act()

# Sample actions from the model.
loss = model.compute_loss(key, obs, act)
print("Loss shape:", loss.shape)

#  [markdown]
# Now, we are going to create a data loader and use a real batch of training data to compute the loss.

# 
# Reduce the batch size to reduce memory usage.
config = dataclasses.replace(config, batch_size=2)

# Load a single batch of data. This is the same data that will be used during training.
# NOTE: In order to make this example self-contained, we are skipping the normalization step
# since it requires the normalization statistics to be generated using `compute_norm_stats`.
loader = _data_loader.create_data_loader(config, num_batches=1, skip_norm_stats=True)
obs, act = next(iter(loader))

# Sample actions from the model.
loss = model.compute_loss(key, obs, act)

# Delete the model to free up memory.
del model

print("Loss shape:", loss.shape)


