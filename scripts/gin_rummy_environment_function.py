import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


GIN_RUMMY_SYSTEM_PROMPT = '''You are playing Gin Rummy.

# Game Rules
GIN RUMMY RULES:

SETUP:
- 52-card deck, each player receives 7-10 cards (variant dependent)
- Goal: Form MELDS to minimize DEADWOOD (unmelded cards)

MELDS (Valid Combinations):
1. SET: 3+ cards of SAME RANK (e.g., 7♠ 7♥ 7♣)
2. RUN: 3+ CONSECUTIVE cards of SAME SUIT (e.g., 5♦ 6♦ 7♦)
Examples:
- Valid runs: A♠-2♠-3♠, 9♥-10♥-J♥-Q♥, 10♣-J♣-Q♣-K♣
- Invalid: K♠-A♠-2♠ (Ace is LOW only, not wraparound)

CARD NOTATION:
- Ranks: A(Ace), 2-9, T(10), J(Jack), Q(Queen), K(King)
- Suits: s(spades♠), h(hearts♥), d(diamonds♦), c(clubs♣)
- Example: 7c = 7 of clubs, Th = 10 of hearts, As = Ace of spades

GAME PHASES:
1. FirstUpcard: Choose to draw first upcard or pass (action IDs: 52=Draw upcard, 54=Pass)
2. Draw: Choose to draw from upcard or stock pile (action IDs: 52=Draw upcard, 53=Draw stock)
3. Discard: Choose which card to discard (action ID = card's index number, shown in Legal Actions)
4. Layoff: After opponent knocks, add cards to their melds or pass (action IDs: card indices or 54=Pass)
5. Knock: Declare end of hand when deadwood ≤ knock_card value

EACH TURN:
1. DRAW phase: Pick from stock pile (53) OR discard pile upcard (52)
2. DISCARD phase: Choose ONE card from hand to discard (use card's action ID from Legal Actions)

KNOCKING:
- When deadwood ≤ knock_card value (8-10), you MAY knock to end hand
- Gin: ALL cards form melds (0 deadwood) = 25-point bonus

SCORING: Winner scores difference in deadwood point values.
Card Values: A=1, 2-10=face value, J=11, Q=12, K=13


# Output Format
First, reason about your strategy inside <think> tags. Then, provide your action ID inside <answer> tags.

<think>strategy reasoning here</think><answer>ACTION_ID</answer>

Examples:
- If the legal actions are "52 -> Draw upcard, 53 -> Draw stock" and you want to draw from stock: <think>The upcard doesn't help my hand. Drawing from stock gives me a chance at a better card. The action id for drawing from stock is 53.</think><answer>53</answer>
- If the legal actions are "0 -> Discard As, 5 -> Discard 6h, 12 -> Discard Kd" and you want to discard the King of diamonds: <think>The King has high deadwood value and doesn't fit any of my melds. The action id for discarding Kd is 12.</think><answer>12</answer>
'''.strip()


MAX_EPISODE_TOKENS = 16384  # Max tokens for completion sequence (truncate if exceeded)
MAX_PROMPT_LEN = 24576      # Max prompt tokens before ending episode early
REQUEST_TIMEOUT = 2400
MAX_PARALLEL_REQUESTS = 10
SELECTED_GAME = "gin_rummy"

GAMES_TO_TASK_ID_RANGE = {
    "goofspiel": (0, 99999999),
    "liars_dice": (100000000, 199999999),
    "leduc_poker": (200000000, 299999999),
    "gin_rummy": (300000000, 399999999),
    "othello": (400000000, 499999999),
    "backgammon": (500000000, 599999999),
    "hex": (600000000, 699999999),
    "clobber": (700000000, 799999999),
}


def format_gin_rummy_observation(raw_obs: str) -> str:
    """
    Parse the raw Gin Rummy observation from the environment and return a
    normalized format. Currently passthrough — game-specific parsing can be
    added later when we see actual API observations.
    """
    try:
        return raw_obs
    except Exception:
        return raw_obs


def extract_action_to_send(completion_text: str) -> str:
    text = completion_text.strip()
    if text.endswith("</s>"):
        text = text[:-5].strip()
    answer_match = re.search(r"<answer>\s*(\d+)\s*</answer>", text)
    if answer_match:
        return answer_match.group(1)

    # Fallback: first integer in text
    fallback_match = re.search(r"\d+", text)
    return fallback_match.group(0) if fallback_match else text


def log_episode_turns(messages: list[dict], episode_index: int, episode_id: str) -> None:
    """
    For a single complete episode consisting of multiple turns, write out
    in plain text what the prompt text and completion text of each turn are
    to a log file.
    """
    try:
        if not messages:
            return

        log_path = os.environ.get(
            "GIN_RUMMY_EPISODE_LOG_PATH",
            "gin_rummy_episode_turns.log",
        )

        lines: list[str] = []
        lines.append(
            f"=== Episode index {episode_index}, episode_id={episode_id} ==="
        )

        turn_id = 0
        i = 0
        while i < len(messages):
            msg = messages[i]
            if (
                msg.get("role") == "user"
                and i + 1 < len(messages)
                and messages[i + 1].get("role") == "assistant"
            ):
                turn_id += 1
                prompt_text = msg.get("content", "")
                completion_text = messages[i + 1].get("content", "")

                lines.append(f"Turn {turn_id} - Prompt:")
                lines.append(prompt_text)
                lines.append("Completion:")
                lines.append(completion_text)
                lines.append("---")
                i += 2
            else:
                i += 1

        lines.append("")  # trailing newline

        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(
            f"[Gin Rummy] Failed to write episode turn log for episode {episode_index}: {e}"
        )


def log_turn(episode_index: int, turn: int, observation: str, completion_text: str, action_to_send: str) -> None:
    try:
        log_path = os.environ.get("GIN_RUMMY_LOG_PATH", "gin_rummy_episodes.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"[Episode {episode_index}] Turn {turn}: prompt={observation[:120]}... "
                f"| completion={completion_text} | action={action_to_send}\n"
            )
    except Exception:
        pass


def initialize_env_endpoint() -> str:
    if not getattr(rollout_first_prompt_and_completion, "initialized", False):
        rank = int(os.environ.get("LOCAL_RANK", "0"))

        raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
        server_list = [url.strip() for url in raw_urls.split(",") if url.strip()]

        if not server_list:
            base_url = ""
            print("Warning: No ENVIRONMENT_SERVER_URLS found.")
        else:
            base_url = server_list[rank % len(server_list)]

        rollout_first_prompt_and_completion.base_url = base_url
        rollout_first_prompt_and_completion.initialized = True
        print(f"Gin Rummy endpoint initialized on rank {rank} at {base_url}")

    return rollout_first_prompt_and_completion.base_url


def reset_episode(i: int, game_id: int, env_endpoint: str) -> tuple[int, str, str, list[dict], bool]:
    payload = {"task_id": game_id, "seed": i, "opponent": "mcts", "mcts_max_simulations": 25, "mcts_num_rollouts": 1}
    try:
        reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=REQUEST_TIMEOUT)
        reset_res.raise_for_status()
        reset_data = reset_res.json()
        result_block = reset_data["result"]

        episode_id = result_block.get("episode_id", "")

        raw_observation = result_block.get("observation", "")
        base_observation = format_gin_rummy_observation(raw_observation)
        observation = f"{GIN_RUMMY_SYSTEM_PROMPT}\n\n{base_observation}"

        user_msg = {"role": "user", "content": observation}
        messages = [user_msg]

        return i, episode_id, observation, messages, False
    except Exception as e:
        print(f"Failed to reset gin rummy game {game_id}: {e}")
        return i, "", "", [], True


def step_episode(
    i: int,
    completion_text: str,
    action_to_send: str,
    episode_id: str,
    env_endpoint: str,
) -> tuple[int, str, str, float, bool]:
    try:
        step_payload = {"action": action_to_send, "episode_id": episode_id}
        step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=REQUEST_TIMEOUT)
        step_res.raise_for_status()
        step_data = step_res.json()
        step_block = step_data["result"]

        raw_step_state = step_block.get("observation", "")
        step_state = format_gin_rummy_observation(raw_step_state)
        step_reward = step_block.get("reward", 0)
        step_done = step_block.get("done", False)

        return i, completion_text, step_state, step_reward, step_done
    except Exception as e:
        print(f"Step failed for gin rummy episode {i}: {e}")
        return i, completion_text, "", -0.01, True


def rollout_first_prompt_and_completion(prompts: list[str], trainer, max_turns: int = 50) -> dict[str, list]:
    from trl.experimental.openenv import generate_rollout_completions
    env_endpoint = initialize_env_endpoint()
    tokenizer = trainer.processing_class
    num_episodes = len(prompts)

    # --- 2. Game ID Assignment ---
    game_ids = [
        random.randint(
            GAMES_TO_TASK_ID_RANGE[SELECTED_GAME][0],
            GAMES_TO_TASK_ID_RANGE[SELECTED_GAME][1],
        )
        for _ in range(num_episodes)
    ]
    episode_ids = [None] * num_episodes
    
    # --- 3. Per-Episode State Tracking ---
    current_observations = ["" for _ in range(num_episodes)]
    done_flags = [False for _ in range(num_episodes)]
    train_rewards = [0.0 for _ in range(num_episodes)]
    episode_lengths = [0 for _ in range(num_episodes)]
    
    # Multi-step accumulator variables (accumulate across all turns)
    accumulated_messages = [[] for _ in range(num_episodes)]
    episode_prompt_ids = [[] for _ in range(num_episodes)]
    episode_completion_ids = [[] for _ in range(num_episodes)]
    episode_logprobs = [[] for _ in range(num_episodes)]
    episode_action_masks = [[] for _ in range(num_episodes)]
    prev_full_ids = [None for _ in range(num_episodes)]

    # Win rate tracking
    wins = 0
    losses = 0
    draws = 0
    completed_games = 0

    # --- 4. Reset All Games (Parallel) ---
    # Execute resets in parallel
    with ThreadPoolExecutor(max_workers=min(num_episodes, MAX_PARALLEL_REQUESTS)) as executor:
        futures = [
            executor.submit(reset_episode, i, game_ids[i], env_endpoint)
            for i in range(num_episodes)
        ]
        for future in as_completed(futures):
            i, episode_id, observation, messages, failed = future.result()
            episode_ids[i] = episode_id
            current_observations[i] = observation
            accumulated_messages[i] = messages
            done_flags[i] = failed

    # --- 5. Batched Turn Loop ---
    for turn in range(max_turns):
        active_indices = [i for i in range(num_episodes) if not done_flags[i]]
        if not active_indices:
            break

        # Build prompts for all active episodes
        batch_prompts = []
        for i in active_indices:
            batch_prompts.append(accumulated_messages[i])

        # --- BATCHED GENERATION ---
        # Generate completions for all active episodes at once
        rollout_outputs = generate_rollout_completions(trainer, prompts=batch_prompts, as_chat=True)

        # Process outputs - extract completions and parse actions
        episode_data = []
        for idx, i in enumerate(active_indices):
            output = rollout_outputs[idx]
            prompt_ids = output.get("prompt_ids", [])
            completion_ids = output.get("completion_ids", [])
            logprobs = output.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            # Check if prompt exceeds max length - end episode early to prevent context overflow
            if len(prompt_ids) > MAX_PROMPT_LEN:
                print(f"Warning: Prompt exceeded {MAX_PROMPT_LEN} tokens ({len(prompt_ids)}) at turn {turn} for episode {i}, ending episode early")
                done_flags[i] = True
                continue

            # Multi-step accumulation logic (from alfworld_legacy.py)
            if turn == 0:
                episode_prompt_ids[i] = prompt_ids
                prev_full_ids[i] = prompt_ids.copy()
            else:
                if prev_full_ids[i] is None:
                    prev_full_ids[i] = prompt_ids.copy()
                else:
                    if prompt_ids[: len(prev_full_ids[i])] != prev_full_ids[i]:
                        num_diff = sum(x != y for x, y in zip(prompt_ids[: len(prev_full_ids[i])], prev_full_ids[i]))
                        print(f"Warning: BPE mismatch at turn {turn} for episode {i}: {num_diff} tokens differ, ratio {100 * num_diff / len(prev_full_ids[i]):.2f}%")
                    delta_prompt_ids = prompt_ids[len(prev_full_ids[i]) :]
                    if delta_prompt_ids:
                        episode_completion_ids[i].extend(delta_prompt_ids)
                        episode_logprobs[i].extend([0.0] * len(delta_prompt_ids))
                        episode_action_masks[i].extend([0] * len(delta_prompt_ids))
                    prev_full_ids[i] = prompt_ids.copy()

            if completion_ids:
                episode_completion_ids[i].extend(completion_ids)
                episode_logprobs[i].extend(logprobs)
                episode_action_masks[i].extend([1] * len(completion_ids))
                if prev_full_ids[i] is not None:
                    prev_full_ids[i] = prev_full_ids[i] + completion_ids

            # --- Parse Action ---
            action_to_send = extract_action_to_send(completion_text)

            # Keep track of observation used to choose this action for optional reward shaping
            obs_before = current_observations[i]

            # --- Per-turn episode logging ---
            log_turn(i, turn, obs_before, completion_text, action_to_send)

            episode_data.append((i, completion_text, action_to_send, obs_before))

        # --- Step Environments (Parallel) ---
        # Execute steps in parallel
        with ThreadPoolExecutor(max_workers=min(len(episode_data), MAX_PARALLEL_REQUESTS)) as executor:
            futures = [
                executor.submit(
                    step_episode,
                    i,
                    comp_text,
                    action,
                    episode_ids[i],
                    env_endpoint,
                )
                for i, comp_text, action, obs_before in episode_data
            ]
            for future in as_completed(futures):
                i, completion_text, step_state, step_reward, step_done = future.result()

                current_observations[i] = step_state
                done_flags[i] = step_done

                # Track episode length for debugging/metrics.
                episode_lengths[i] += 1

                if step_done:
                    train_rewards[i] = step_reward
                    completed_games += 1
                    if step_reward > 0.5:
                        wins += 1
                    elif step_reward < 0.5:
                        losses += 1
                    else:
                        draws += 1

                # Update messages for next turn
                assistant_msg = {"role": "assistant", "content": completion_text}
                accumulated_messages[i].append(assistant_msg)

                if not step_done:
                    user_msg = {"role": "user", "content": step_state}
                    accumulated_messages[i].append(user_msg)
                else:
                    log_episode_turns(
                        accumulated_messages[i],
                        i,
                        episode_ids[i],
                    )

    # --- 6. Log win rate to WandB and stdout ---
    win_rate = wins / max(completed_games, 1)

    print(f"[Gin Rummy] Batch stats: {wins}W/{losses}L/{draws}D "
          f"({completed_games} games, win_rate={win_rate:.2%})")

    try:
        import wandb
        if wandb.run is not None:
            wandb.log({
                "game/win_rate": win_rate,
                "game/wins": wins,
                "game/losses": losses,
                "game/draws": draws,
                "game/completed_games": completed_games,
            }, commit=False)
    except ImportError:
        pass

    # --- 7. Return Results ---
    all_prompt_ids = []
    all_completion_ids = []
    all_logprobs = []
    all_rewards = []
    all_action_masks = []

    for i in range(num_episodes):
        # Skip episodes with empty sequences
        if not episode_prompt_ids[i] or not episode_completion_ids[i]:
            continue

        # Truncate episode if completion sequence exceeds max length
        if len(episode_completion_ids[i]) > MAX_EPISODE_TOKENS:
            print(f"Warning: Episode {i} completion exceeded {MAX_EPISODE_TOKENS} tokens ({len(episode_completion_ids[i])}), truncating")
            episode_completion_ids[i] = episode_completion_ids[i][:MAX_EPISODE_TOKENS]
            episode_logprobs[i] = episode_logprobs[i][:MAX_EPISODE_TOKENS]
            episode_action_masks[i] = episode_action_masks[i][:MAX_EPISODE_TOKENS]

        all_prompt_ids.append(episode_prompt_ids[i])
        all_completion_ids.append(episode_completion_ids[i])
        all_logprobs.append(episode_logprobs[i])
        all_rewards.append(train_rewards[i])
        all_action_masks.append(episode_action_masks[i])

    return {
        "prompt_ids": all_prompt_ids,
        "completion_ids": all_completion_ids,
        "logprobs": all_logprobs,
        "env_rewards": all_rewards,
        "action_mask": all_action_masks,
    }

def rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)