import gymnasium as gym

gym.register(
    id="FrankaCubeStack-v0",
    entry_point="project.envs.franka_cube_stack:FrankaCubeStackEnv",
    max_episode_steps=1000,
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
