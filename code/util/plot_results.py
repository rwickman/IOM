import matplotlib.pyplot as plt
import numpy as np
import json

policy_dir = "/media/data/code/IOM/eval_policies_gamma995_2/dqn_emb_2/train_dict.json"
with open(policy_dir) as f:
    train_dict = json.load(f)

"""Plot the training results."""
def moving_average(x, w=64):
    return np.convolve(x, np.ones(w), 'valid') / w

fig, axs = plt.subplots(2)
y = moving_average(train_dict["ep_avg_rewards"])
axs[0].plot(y)
axs[0].set(
    title="Episode Average Reward",
    xlabel="Episode",
    ylabel="Average Reward"
)

axs[1].plot(train_dict["policy_losses"])
axs[1].set(
    title="Policy MSE Loss",
    xlabel="Episode",
    ylabel="Average Loss"
)

plt.show()