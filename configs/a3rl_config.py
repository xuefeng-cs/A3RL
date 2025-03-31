from configs import sac_priority_config
# eventually sac_priority_config once SACLearnerPriority works

def get_config():
    config = sac_priority_config.get_config()

    config.num_qs = 10
    config.num_min_qs = 2
    config.critic_layer_norm = True

    return config
