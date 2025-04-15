import gymnasium as gym

gym.register(
    id="FrankaCubePick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    # max_episode_steps=1000,
    kwargs={
        "env_config": {
            "scene": {
                "num_envs": 1,
                "device": "cuda",
                "headless": False,
            },
            "sim": {
                "device": "cuda",
            },
        }
    },
)
