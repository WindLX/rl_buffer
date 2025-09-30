import torch
from rl_buffer import ReplayBuffer, StatsTracker


def run_replay_buffer_example():
    """
    Demonstrates the usage of ReplayBuffer for off-policy algorithms.
    """
    # 1. Environment and buffer configuration
    NUM_ENVS = 4
    BUFFER_SIZE = 1000
    BATCH_SIZE = 64
    OBS_SHAPE = (16,)
    ACTION_SHAPE = (4,)
    # Use torch.uint8 for observations to demonstrate dtype flexibility
    OBS_DTYPE = torch.uint8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Initialize StatsTracker and ReplayBuffer
    stats_tracker = StatsTracker(num_envs=NUM_ENVS, category="OffPolicy")
    replay_buffer = ReplayBuffer(
        buffer_size=BUFFER_SIZE,
        num_envs=NUM_ENVS,
        obs_shape=OBS_SHAPE,
        action_shape=ACTION_SHAPE,
        obs_dtype=OBS_DTYPE,
        stats_tracker=stats_tracker,
        device=DEVICE,
    )

    print(f"--- Populating ReplayBuffer on {DEVICE} ---")
    print(f"Observation dtype: {replay_buffer._observations.dtype}")

    # 3. Simulate a data collection loop
    for step in range(BUFFER_SIZE):
        # Generate dummy data
        # Observations are integers in [0, 255] for uint8
        obs = torch.randint(0, 256, (NUM_ENVS, *OBS_SHAPE), dtype=OBS_DTYPE)
        action = torch.randn(NUM_ENVS, *ACTION_SHAPE)
        reward = torch.randn(NUM_ENVS)
        # Simulate that an environment terminates roughly every 100 steps
        done = torch.rand(NUM_ENVS) < 0.01
        next_obs = torch.randint(0, 256, (NUM_ENVS, *OBS_SHAPE), dtype=OBS_DTYPE)

        # Add data to the buffer
        replay_buffer.add(obs, action, reward, done, next_obs)

    print(f"Buffer populated. Current size: {len(replay_buffer)}")

    # 4. Sample a batch from the buffer for training
    if len(replay_buffer) >= BATCH_SIZE:
        print(f"\n--- Sampling a batch of size {BATCH_SIZE} ---")
        batch = replay_buffer.get(batch_size=BATCH_SIZE)

        # In a real training loop, you'd normalize observations if they are uint8
        # obs_float = batch.observations.float() / 255.0

        print(
            f"Sampled observations shape: {batch.observations.shape} (dtype: {batch.observations.dtype})"
        )
        print(f"Sampled actions shape: {batch.actions.shape}")
        print(f"Sampled rewards shape: {batch.rewards.shape}")

    # 5. Get and print the collected statistics
    print("\n--- Episode Statistics ---")
    stats = stats_tracker.get_statistics()
    if not stats:
        print("No complete episodes during this short rollout.")
    else:
        for key, value in stats.items():
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    run_replay_buffer_example()
