class FakeArgs:
    def __init__(self):
        self.reward_alpha = 0.5
        self.gamma = 0.99
        self.hidden_size = 8
        self.num_inv_nodes = 2
        self.num_skus = 2
        self.save_dir = "."
        self.lr = 1e-4
        self.no_per = True
        self.load = False
        self.min_epsilon = 0.05
        self.epsilon = 0.5
        self.epsilon_decay = 1024
        self.mem_cap = 3
        self.per_beta = 0.4
        self.per_alpha = 0.6
        self.eps = 1e-6
        self.coord_bounds = 10
        self.min_inv_prod = 1
        self.max_inv_prod = 1
        self.kappa = 2
        self.loc_json = "fake_loc.json"
        self.policy = "naive"
        self.policy_dir = "fake_policy_dir"
        self.eval_episodes = 2
        self.critic_lam = 0.5
        self.gae_lam = 0.95
