import torch
from rl_buffer import RolloutBuffer, StatsTracker


def run_rollout_buffer_example():
    """
    Demonstrates the usage of RolloutBuffer for on-policy algorithms like PPO.
    """
    # 1. Environment and buffer configuration
    NUM_ENVS = 4
    BUFFER_SIZE = 256  # In on-policy, this is often called n_steps
    BATCH_SIZE = 64
    N_EPOCHS = 4
    OBS_SHAPE = (16,)
    ACTION_SHAPE = (1,)  # e.g., for discrete actions
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. Initialize StatsTracker and RolloutBuffer
    stats_tracker = StatsTracker(num_envs=NUM_ENVS, category="OnPolicy")
    rollout_buffer = RolloutBuffer(
        buffer_size=BUFFER_SIZE,
        num_envs=NUM_ENVS,
        obs_shape=OBS_SHAPE,
        action_shape=ACTION_SHAPE,
        stats_tracker=stats_tracker,
        device=DEVICE,
        gamma=0.99,
        gae_lambda=0.95,
    )

    print(f"--- Collecting one rollout of {BUFFER_SIZE} steps on {DEVICE} ---")

    # 3. Simulate the data collection phase (one full rollout)
    for _ in range(BUFFER_SIZE):
        # Generate dummy data from a "policy"
        obs = torch.randn(NUM_ENVS, *OBS_SHAPE)
        action = torch.randint(0, 10, (NUM_ENVS, *ACTION_SHAPE))  # discrete actions
        reward = torch.rand(NUM_ENVS)
        done = torch.zeros(NUM_ENVS, dtype=torch.bool)
        value = torch.randn(NUM_ENVS)  # from critic
        log_prob = -torch.rand(NUM_ENVS)  # from actor

        # Add data to the buffer
        rollout_buffer.add(obs, action, reward, done, value, log_prob)

    print("Rollout collection complete.")

    # 4. Compute advantages and returns after the rollout
    # In a real loop, you'd get the value estimate for the last observation
    with torch.no_grad():
        last_values = torch.randn(NUM_ENVS)  # critic(last_obs)
        last_dones = torch.zeros(NUM_ENVS, dtype=torch.bool)

    rollout_buffer.compute_returns_and_advantages(last_values, last_dones)
    print("GAE and returns have been computed.")

    print(f"\n--- Simulating training for {N_EPOCHS} epochs ---")

    # 5. Use the buffer's generator to iterate over mini-batches for multiple epochs
    for epoch in range(N_EPOCHS):
        batch_count = 0
        # The .get() method returns a new shuffled generator each time it's called
        for batch in rollout_buffer.get(batch_size=BATCH_SIZE):
            # In a real PPO, you would perform a gradient update here:
            # actor_loss, critic_loss = compute_ppo_loss(batch)
            # optimizer.zero_grad()
            # (actor_loss + critic_loss).backward()
            # optimizer.step()
            batch_count += 1

        print(f"Epoch {epoch + 1}: Trained on {batch_count} mini-batches.")

    # 6. Get and print the collected statistics
    print("\n--- Episode Statistics ---")
    stats = stats_tracker.get_statistics()
    if not stats:
        print("No complete episodes during this short rollout.")
    else:
        for key, value in stats.items():
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    run_rollout_buffer_example()
