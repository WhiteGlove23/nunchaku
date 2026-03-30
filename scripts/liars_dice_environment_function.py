import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from threading import Lock, Semaphore

import requests
from trl.experimental.openenv import generate_rollout_completions

GAME_TO_TASK_ID_RANGE = {
    "goofspiel": (0, 99999999),
    "liars_dice": (100000000, 199999999),
    "leduc_poker": (200000000, 299999999),
    "gin_rummy": (300000000, 399999999),
    "othello": (400000000, 499999999),
    "backgammon": (500000000, 599999999),
    "hex": (600000000, 699999999),
    "clobber": (700000000, 799999999),
}

SELECTED_GAME = "liars_dice"
REQUEST_TIMEOUT_SECONDS = 2400
INIT_TIMEOUT_SECONDS = 300
MAX_EPISODE_TOKENS = 16384
MAX_PROMPT_LEN = 16384 - 512

MCTS_CONFIG = {
    "opponent": "mcts",
    "mcts_max_simulations":50,
    "mcts_num_rollouts": 1,
}

# Reward settings
INVALID_ACTION_PENALTY = 0.10
RULE_ACTION_MATCH_REWARD = 0.20
RULE_ACTION_MISMATCH_PENALTY = 0.20
RULE_INVALID_ACTION_PENALTY = 0.30
IMITATION_REWARD_CLIP = 1.00
TERMINAL_REWARD_CLIP = 1.00
TERMINAL_REWARD_WEIGHT = 0.10

RULE_POLICY_MODE = "probability"
RULE_LIAR_PROB_THRESHOLD = 0.30
RULE_LIAR_FACE5_THRESHOLD = 0.30
RULE_LIAR_FACE6_THRESHOLD = 0.30
RULE_OPENING_MODE = "best_support_low"
RULE_OPENING_MAX_QUANTITY = 2
RULE_RAISE_MODE = "max_support_then_low"

RULE_OPPONENT_TRUTH_PROBABILITY = {
    6: {
        0: 1.0,
        1: 0.5981224279835391,
        2: 0.1962448559670782,
        3: 0.035493827160493825,
        4: 0.003343621399176955,
        5: 0.0001286008230452675,
    },
    5: {
        0: 1.0,
        1: 0.8683127572016461,
        2: 0.5390946502057613,
        3: 0.20987654320987653,
        4: 0.04526748971193416,
        5: 0.00411522633744856,
    },
}

STRATEGY_TIPS = """
STRATEGY TIPS:
- Output only the action ID. Never include explanations or extra text.
- Treat 6s as wild only for bids on faces 1-5. For bids on face 6, count only actual 6s.
- If the current bid's rule truth probability is 0.30 or lower, choose Liar immediately.
- If there is no current bid, open with the best-supported legal bid of quantity 2 or lower.
- Otherwise raise using this exact priority: highest own support, then lowest bid rank, then highest truth probability.
- When multiple actions seem close, follow the exact tie-break rules above rather than a general strategy.
"""

REASONING_TAG_PAIRS = [
    ("think", "think"),
    ("thinking", "thinking"),
    ("reasoning", "reasoning"),
    ("thought", "thought"),
    ("reflection", "reflection"),
]

_ROLLOUT_STATE: dict = {}


def _is_truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def extract_and_format_observation(obs_text: str) -> str:
    # Liar's Dice observations already contain structured legal-action blocks.
    return obs_text or ""


class EpisodeTraceLogger:
    """Thread-safe JSONL episode tracer."""

    def __init__(self, trace_dir: str, rank: int):
        self.trace_dir = trace_dir
        self.rank = rank
        self._lock = Lock()
        self.log_path = os.path.join(self.trace_dir, f"liars_dice_episode_traces_rank{rank}.jsonl")
        self.max_text_chars = int(os.environ.get("EPISODE_TRACE_MAX_TEXT_CHARS", "4000"))
        self.sample_rate = float(os.environ.get("EPISODE_TRACE_SAMPLE_RATE", "1.0"))

        os.makedirs(self.trace_dir, exist_ok=True)
        print(f"[EPISODE_TRACE] Writing traces to {self.log_path}")

    def should_log(self) -> bool:
        if self.sample_rate >= 1.0:
            return True
        if self.sample_rate <= 0.0:
            return False
        return random.random() <= self.sample_rate

    def clip_text(self, text: str) -> str:
        if not text:
            return ""
        if len(text) <= self.max_text_chars:
            return text
        return text[: self.max_text_chars] + f"... [truncated {len(text) - self.max_text_chars} chars]"

    def log_episode(self, payload: dict) -> None:
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True) + "\n")


class CurriculumScheduler:
    """Progressive turn-limit curriculum."""

    def __init__(
        self,
        initial_max_turn: int = 2,
        final_max_turn: int = 20,
        rollouts_per_stage: int = 1280,
        initial_hint_prob: float = 0.0,
        final_hint_prob: float = 0.0,
        warmup_rollouts: int = 128,
    ):
        self.initial_max_turn = initial_max_turn
        self.final_max_turn = final_max_turn
        self.rollouts_per_stage = rollouts_per_stage
        self.initial_hint_prob = initial_hint_prob
        self.final_hint_prob = final_hint_prob
        self.warmup_rollouts = warmup_rollouts
        self.total_rollouts = 0

    def get_max_turn(self) -> int:
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_max_turn
        adjusted_rollouts = self.total_rollouts - self.warmup_rollouts
        stage = adjusted_rollouts // self.rollouts_per_stage
        return min(self.initial_max_turn + stage, self.final_max_turn)

    def get_hint_prob(self) -> float:
        if self.total_rollouts < self.warmup_rollouts:
            return self.initial_hint_prob
        total_stages = max(self.final_max_turn - self.initial_max_turn, 1)
        total_decay_rollouts = total_stages * self.rollouts_per_stage
        adjusted_rollouts = self.total_rollouts - self.warmup_rollouts
        progress = min(adjusted_rollouts / total_decay_rollouts, 1.0)
        current_prob = self.initial_hint_prob - progress * (self.initial_hint_prob - self.final_hint_prob)
        return max(current_prob, self.final_hint_prob)

    def step(self, num_rollouts: int = 1) -> None:
        self.total_rollouts += num_rollouts


def remove_reasoning_tags(text: str) -> str:
    cleaned = text
    for tag_name, close_name in REASONING_TAG_PAIRS:
        cleaned = re.sub(
            rf"<{tag_name}>.*?</{close_name}>",
            "",
            cleaned,
            flags=re.DOTALL | re.IGNORECASE,
        )
        close_tag = f"</{close_name}>"
        if close_tag in cleaned:
            cleaned = cleaned.split(close_tag)[-1]
        open_match = re.search(rf"<{tag_name}>", cleaned, flags=re.IGNORECASE)
        if open_match:
            cleaned = cleaned[: open_match.start()]
    cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
    return cleaned.strip()


def _extract_legal_action_map(observation: str) -> dict[str, str]:
    if not observation:
        return {}
    match = re.search(
        r"Legal Actions:\s*\n(.*?)(?:\n\nYour choice|\nYour choice|\Z)",
        observation,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return {}

    block = match.group(1)
    mapping: dict[str, str] = {}
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "->" in line:
            left, right = line.split("->", 1)
            action_id = left.strip()
            label = right.strip()
        else:
            action_id = line.strip()
            label = action_id
        if re.fullmatch(r"-?\d+", action_id):
            mapping[action_id] = label
    return mapping


def _extract_bid_tuple(label_or_text: str) -> tuple[int, int] | None:
    if not label_or_text:
        return None
    match = re.search(r"(\d+)\s*-\s*(\d+)", label_or_text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _extract_state_features(observation: str) -> dict:
    dice: list[int] = []
    dice_match = re.search(r"Your dice:\s*\[([^\]]*)\]", observation)
    if dice_match:
        dice_str = dice_match.group(1).strip()
        if dice_str:
            dice = [int(x.strip()) for x in dice_str.split(",") if x.strip().isdigit()]

    total_dice_match = re.search(r"Total dice in game:\s*(\d+)", observation)
    total_dice = int(total_dice_match.group(1)) if total_dice_match else 0

    current_bid_match = re.search(r'Current bid:\s*"([^"]+)"', observation)
    current_bid = _extract_bid_tuple(current_bid_match.group(1)) if current_bid_match else None

    return {
        "own_dice": dice,
        "total_dice": total_dice,
        "current_bid": current_bid,
        "wild_six_enabled": "wild" in observation.lower() and "6" in observation,
    }


def _is_liar_label(label: str) -> bool:
    return "liar" in (label or "").strip().lower()


def _bid_rank(bid: tuple[int, int]) -> int:
    quantity, face = bid
    return quantity * 6 + face


def _rule_count_face_support(own_dice: list[int], target_face: int) -> int:
    if target_face == 6:
        return sum(1 for value in own_dice if value == 6)
    return sum(1 for value in own_dice if value == target_face or value == 6)


def _rule_bid_truth_probability(own_dice: list[int], bid: tuple[int, int]) -> float:
    quantity, face = bid
    own_support = _rule_count_face_support(own_dice, face)
    need_from_opponent = max(0, quantity - own_support)
    table_key = 6 if face == 6 else 5
    return float(RULE_OPPONENT_TRUTH_PROBABILITY[table_key].get(need_from_opponent, 0.0))


def _rule_liar_threshold_for_bid(bid: tuple[int, int]) -> float:
    _, face = bid
    if face == 6:
        return RULE_LIAR_FACE6_THRESHOLD
    if face == 5:
        return RULE_LIAR_FACE5_THRESHOLD
    return RULE_LIAR_PROB_THRESHOLD


def _rule_choose_raise_action(legal_action_map: dict[str, str], own_dice: list[int], mode: str) -> tuple[str, dict]:
    scored_actions = []
    for action_id, label in legal_action_map.items():
        if _is_liar_label(label):
            continue
        bid = _extract_bid_tuple(label)
        if bid is None:
            continue
        truth_probability = _rule_bid_truth_probability(own_dice, bid)
        own_support = _rule_count_face_support(own_dice, bid[1])
        scored_actions.append((action_id, bid, truth_probability, own_support))

    if not scored_actions:
        return "", {}

    if mode == "max_truth_then_low":
        scored_actions.sort(key=lambda item: (-item[2], _bid_rank(item[1]), -item[3]))
    elif mode == "max_support_then_low":
        scored_actions.sort(key=lambda item: (-item[3], _bid_rank(item[1]), -item[2]))
    else:
        scored_actions.sort(key=lambda item: _bid_rank(item[1]))

    action_id, bid, truth_probability, own_support = scored_actions[0]
    return action_id, {
        "selected_bid": bid,
        "selected_bid_support": own_support,
        "selected_bid_truth_probability": truth_probability,
    }


def _rule_choose_opening_action(legal_action_map: dict[str, str], own_dice: list[int]) -> tuple[str, dict]:
    scored_actions = []
    for action_id, label in legal_action_map.items():
        bid = _extract_bid_tuple(label)
        if bid is None or bid[0] > RULE_OPENING_MAX_QUANTITY:
            continue
        truth_probability = _rule_bid_truth_probability(own_dice, bid)
        own_support = _rule_count_face_support(own_dice, bid[1])
        scored_actions.append((action_id, bid, truth_probability, own_support))

    if not scored_actions:
        return "", {}

    if RULE_OPENING_MODE == "best_support_low":
        scored_actions.sort(key=lambda item: (-item[3], _bid_rank(item[1]), -item[2]))
    elif RULE_OPENING_MODE == "best_truth_low":
        scored_actions.sort(key=lambda item: (-item[2], _bid_rank(item[1]), -item[3]))
    else:
        scored_actions.sort(key=lambda item: _bid_rank(item[1]))

    action_id, bid, truth_probability, own_support = scored_actions[0]
    return action_id, {
        "selected_bid": bid,
        "selected_bid_support": own_support,
        "selected_bid_truth_probability": truth_probability,
    }


def _select_rule_target_action(observation: str, legal_action_map: dict[str, str]) -> tuple[str, dict]:
    if not legal_action_map:
        return "", {}

    state_features = _extract_state_features(observation)
    current_bid = state_features.get("current_bid")
    own_dice = state_features.get("own_dice") or []
    liar_action_id = next(
        (action_id for action_id, label in legal_action_map.items() if _is_liar_label(label)),
        "",
    )

    if RULE_POLICY_MODE == "probability":
        if current_bid is None:
            opening_action, opening_meta = _rule_choose_opening_action(legal_action_map, own_dice)
            if opening_action:
                return opening_action, {
                    "current_bid_probability": 0.0,
                    **opening_meta,
                }
        else:
            current_bid_probability = _rule_bid_truth_probability(own_dice, current_bid)
            if liar_action_id and current_bid_probability <= _rule_liar_threshold_for_bid(current_bid):
                return liar_action_id, {
                    "current_bid_probability": current_bid_probability,
                    "selected_bid": None,
                    "selected_bid_support": 0,
                    "selected_bid_truth_probability": 0.0,
                }
            raise_action, raise_meta = _rule_choose_raise_action(legal_action_map, own_dice, RULE_RAISE_MODE)
            if raise_action:
                return raise_action, {
                    "current_bid_probability": current_bid_probability,
                    **raise_meta,
                }

    fallback_action = sorted(legal_action_map.keys(), key=lambda x: int(x))[0]
    return fallback_action, {
        "current_bid_probability": 0.0,
        "selected_bid": None,
        "selected_bid_support": 0,
        "selected_bid_truth_probability": 0.0,
    }

def _parse_action_id(completion_text: str, legal_action_map: dict[str, str]) -> str:
    if not legal_action_map:
        return ""

    cleaned = remove_reasoning_tags(completion_text or "")
    if cleaned.endswith("</s>"):
        cleaned = cleaned[:-5]
    if "Action:" in cleaned:
        cleaned = cleaned.split("Action:")[-1].strip()

    for num in re.findall(r"-?\d+", cleaned):
        if num in legal_action_map:
            return num

    normalized = cleaned.strip().lower()
    for action_id, label in legal_action_map.items():
        if normalized == label.strip().lower():
            return action_id

    if "liar" in normalized:
        for action_id, label in legal_action_map.items():
            if _is_liar_label(label):
                return action_id

    bid_tuple = _extract_bid_tuple(cleaned)
    if bid_tuple is not None:
        for action_id, label in legal_action_map.items():
            if _extract_bid_tuple(label) == bid_tuple:
                return action_id

    return ""


def _select_fallback_action(legal_action_map: dict[str, str]) -> str:
    return sorted(legal_action_map.keys(), key=lambda x: int(x))[0]


def _extract_terminal_reward(step_block: dict, observation_text: str) -> float:
    info = step_block.get("info", {}) if isinstance(step_block, dict) else {}

    cumulative_reward = info.get("cumulative_reward")
    if isinstance(cumulative_reward, (int, float)):
        return _clamp(float(cumulative_reward), -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)

    your_return_match = re.search(r"Your Return:\s*([+-]?\d+(?:\.\d+)?)", observation_text or "")
    if your_return_match:
        return _clamp(float(your_return_match.group(1)), -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)

    normalized_match = re.search(r"Normalized Score:\s*([+-]?\d+(?:\.\d+)?)", observation_text or "")
    result_match = re.search(r"Result:\s*(WIN|LOSS|DRAW)", observation_text or "", flags=re.IGNORECASE)
    if normalized_match:
        normalized_value = float(normalized_match.group(1))
        if result_match:
            result = result_match.group(1).upper()
            if result == "LOSS":
                normalized_value = -abs(normalized_value) if normalized_value != 0 else -1.0
            elif result == "WIN":
                normalized_value = abs(normalized_value) if normalized_value != 0 else 1.0
            else:
                normalized_value = 0.0
        return _clamp(normalized_value, -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)

    step_reward = _safe_float(step_block.get("reward", 0.0), default=0.0)
    return _clamp(step_reward, -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)


def _build_env_pool(server_urls: list[str]) -> list[dict[str, str]]:
    env_pool = []
    init_task_id = GAME_TO_TASK_ID_RANGE[SELECTED_GAME][0]

    for idx, base_url in enumerate(server_urls):
        try:
            print(f"[INIT] Initializing env on server {idx}: {base_url}")
            payload = {"task_id": init_task_id, "seed": 42, **MCTS_CONFIG}
            res = requests.post(f"{base_url}/reset", json=payload, timeout=INIT_TIMEOUT_SECONDS)
            res.raise_for_status()
            env_pool.append({"base_url": base_url})
            print(f"[INIT] Server {idx} ready")
        except Exception as e:
            raise RuntimeError(f"Failed to init server {base_url}: {e}") from e

    return env_pool


def _initialize_rollout_state(trainer) -> None:
    if _ROLLOUT_STATE.get("initialized", False):
        return

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    raw_urls = os.environ.get("ENVIRONMENT_SERVER_URLS", "")
    server_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]
    if not server_urls:
        raise RuntimeError("ENVIRONMENT_SERVER_URLS is empty")

    env_pool = _build_env_pool(server_urls)
    rollout_per_stage = int(getattr(trainer.args, "rollouts_per_stage", 1280))
    initial_max_turn = int(getattr(trainer.args, "initial_max_turn", 2))
    final_max_turn = int(os.environ.get("LIARS_DICE_FINAL_MAX_TURN", "20"))
    initial_hint_prob = float(os.environ.get("LIARS_DICE_INITIAL_HINT_PROB", "0.0"))
    final_hint_prob = float(os.environ.get("LIARS_DICE_FINAL_HINT_PROB", "0.0"))

    _ROLLOUT_STATE["rank"] = rank
    _ROLLOUT_STATE["env_pool"] = env_pool
    _ROLLOUT_STATE["num_servers"] = len(env_pool)
    _ROLLOUT_STATE["thread_pool"] = ThreadPoolExecutor(max_workers=len(env_pool))
    _ROLLOUT_STATE["generation_semaphore"] = Semaphore(1)
    _ROLLOUT_STATE["curriculum"] = CurriculumScheduler(
        initial_max_turn=initial_max_turn,
        final_max_turn=final_max_turn,
        rollouts_per_stage=rollout_per_stage,
        initial_hint_prob=initial_hint_prob,
        final_hint_prob=final_hint_prob,
        warmup_rollouts=128,
    )
    _ROLLOUT_STATE["initialized"] = True

    trace_enabled = _is_truthy_env(os.environ.get("EPISODE_TRACE_ENABLED", "1"))
    trace_dir = os.environ.get("EPISODE_TRACE_DIR", "").strip()
    _ROLLOUT_STATE["trace_logger"] = None
    if trace_enabled and trace_dir:
        try:
            _ROLLOUT_STATE["trace_logger"] = EpisodeTraceLogger(trace_dir=trace_dir, rank=rank)
        except Exception as e:
            print(f"[EPISODE_TRACE] Failed to initialize logger: {e}")
    elif rank == 0:
        print("[EPISODE_TRACE] Disabled (set EPISODE_TRACE_ENABLED=1 and EPISODE_TRACE_DIR)")


def _reset_environment(env_endpoint: str, game_id: int, timeout: int) -> tuple[str, str]:
    payload = {"task_id": game_id, "seed": random.randint(0, 2**31 - 1), **MCTS_CONFIG}
    reset_res = requests.post(f"{env_endpoint}/reset", json=payload, timeout=timeout)
    reset_res.raise_for_status()
    reset_data = reset_res.json()
    result_block = reset_data["result"]
    episode_id = result_block.get("episode_id", "")
    raw_observation = result_block.get("observation", "")
    return episode_id, extract_and_format_observation(raw_observation)


def _step_environment(
    env_endpoint: str,
    episode_id: str,
    action_to_send: str,
    timeout: int,
) -> tuple[str, float, bool, dict]:
    step_payload = {"action": action_to_send, "episode_id": episode_id}
    step_res = requests.post(f"{env_endpoint}/step", json=step_payload, timeout=timeout)
    step_res.raise_for_status()
    step_data = step_res.json()
    step_block = step_data["result"]
    raw_observation = step_block.get("observation", "")
    formatted_observation = extract_and_format_observation(raw_observation)
    step_reward = _safe_float(step_block.get("reward", 0.0), default=0.0)
    done = bool(step_block.get("done", False))
    return formatted_observation, step_reward, done, step_block


def _last_prompt_fallback_result() -> dict:
    return {
        "prompt_ids": [1],
        "completion_ids": [1],
        "logprobs": [1.0],
        "reward": 0.0,
        "final_score": 0.0,
    }


def _full_prompt_fallback_result() -> dict:
    return {
        "prompt_ids": [1],
        "completion_ids": [1],
        "action_mask": [0],
        "logprobs": [1.0],
        "reward": 0.0,
        "final_score": 0.0,
    }


def _execute_parallel_rollouts(prompts, executor, run_single_prompt, fallback_builder):
    results = [None] * len(prompts)
    futures = [executor.submit(run_single_prompt, i, p) for i, p in enumerate(prompts)]

    for future in as_completed(futures):
        idx, res = future.result()
        results[idx] = res if res is not None else fallback_builder()

    return [r for r in results if r is not None]


def _log_batch_statistics(list_results: list[dict]) -> None:
    finished = sum(1 for r in list_results if r["final_score"] != 0)
    avg_return = sum(r["reward"] for r in list_results) / len(list_results) if list_results else 0.0
    print(f"[BATCH] Finished: {finished}/{len(list_results)}, AvgReturn: {avg_return:.3f}")


def _get_system_prompt(use_hints: bool) -> str:
    system_prompt = """You are playing liars_dice.

# Game Rules
LIAR'S DICE RULES:

Setup: Each player has N dice (1-5 depending on variant). All players roll their dice secretly.

Goal: Make bids about total dice across ALL players, or call "Liar" on opponent's bid.

Actions:
- Bid (quantity, face): Claim there are at least 'quantity' dice showing 'face' among all dice.
- Call Liar: Challenge the previous bid.

Bidding rules: Each bid must be higher than the previous bid. "Higher" means:
  - Same face value but higher quantity (e.g., "2 fours" beats "1 four")
  - Same quantity but higher face value (e.g., "2 fives" beats "2 fours")

Wild dice: 6s are WILD and count as ANY face value.
- When counting dice for a bid, include 6s in the count
- Example: Bid "3 fours" means at least 3 dice showing EITHER 4 OR 6

Winning: If you call Liar and previous bid was false, opponent loses. If bid was true or exact, you lose.

# Output Format
You must respond with ONLY the action ID (a single number). 
Do NOT include descriptions or explanations. 
Examples:
- For action "59 -> 10-6": respond "59"
- For action "60 -> Liar": respond "60"

# Policy To Follow
Follow this exact action-selection policy:
- Count support for bids on faces 1-5 as dice matching that face plus any 6s.
- Count support for bids on face 6 using only actual 6s.
- If there is a current bid and its rule truth probability is 0.30 or lower, choose Liar immediately.
- If there is no current bid, choose the legal opening bid with quantity 2 or lower that maximizes own support, then minimizes bid rank, then maximizes truth probability.
- Otherwise choose the legal raise that maximizes own support, then minimizes bid rank, then maximizes truth probability.
- If multiple actions look plausible, follow the exact tie-break rules above.
"""
    if use_hints:
        system_prompt += "\n" + STRATEGY_TIPS
    return system_prompt


def _rollout_parallelized_curriculum(
    prompts: list[str],
    trainer,
    include_action_mask: bool,
) -> dict[str, list]:
    _initialize_rollout_state(trainer)

    rank = _ROLLOUT_STATE["rank"]
    env_pool = _ROLLOUT_STATE["env_pool"]
    num_servers = _ROLLOUT_STATE["num_servers"]
    curriculum: CurriculumScheduler = _ROLLOUT_STATE["curriculum"]
    trace_logger = _ROLLOUT_STATE["trace_logger"]

    tokenizer = trainer.processing_class
    timeout = REQUEST_TIMEOUT_SECONDS
    current_max_turn = curriculum.get_max_turn()
    current_hint_prob = curriculum.get_hint_prob()
    print(
        f"[CURRICULUM] Rollout {curriculum.total_rollouts}: "
        f"max_turn={current_max_turn}, hint_prob={current_hint_prob:.2f}"
    )

    def run_single_prompt(index: int, prompt: str):
        game_id = int(prompt)
        server_idx = (index + rank) % num_servers
        server = env_pool[server_idx]
        env_endpoint = server["base_url"]

        invalid_count = 0
        done = False
        final_reward = 0.0
        turn_number = 0
        accumulated_imitation_reward = 0.0
        step_records = []
        termination_reason = "unknown"
        last_step_block: dict = {}

        if include_action_mask:
            episode_prompt_ids: list[int] = []
            episode_completion_ids: list[int] = []
            episode_logprobs: list[float] = []
            episode_action_mask: list[int] = []
            prev_full_ids: list[int] | None = None
        else:
            prompt_ids_last: list[int] = []
            completion_ids_last: list[int] = []
            logprobs_last: list[float] = []

        try:
            episode_id, formatted_observation = _reset_environment(
                env_endpoint=env_endpoint,
                game_id=game_id,
                timeout=timeout,
            )
        except Exception as e:
            print(f"Failed to reset environment (Game {game_id}): {e}")
            if trace_logger and trace_logger.should_log():
                trace_logger.log_episode(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "game_id": game_id,
                        "status": "reset_failed",
                        "error": str(e),
                    }
                )
            return index, None

        use_hints = random.random() < current_hint_prob
        messages = [
            {"role": "system", "content": _get_system_prompt(use_hints=use_hints)},
            {"role": "user", "content": formatted_observation},
        ]

        while not done and turn_number < current_max_turn:
            observation_before_action = formatted_observation
            legal_action_map = _extract_legal_action_map(observation_before_action)

            if not legal_action_map:
                accumulated_imitation_reward -= RULE_INVALID_ACTION_PENALTY
                termination_reason = "no_legal_actions"
                break

            target_action_id, target_action_meta = _select_rule_target_action(
                observation=observation_before_action,
                legal_action_map=legal_action_map,
            )
            target_action_label = legal_action_map.get(target_action_id, "")

            with _ROLLOUT_STATE["generation_semaphore"]:
                rollout_outputs = generate_rollout_completions(trainer, prompts=[messages], as_chat=True)[0]

            prompt_ids = rollout_outputs.get("prompt_ids", [])
            completion_ids = rollout_outputs.get("completion_ids", [])
            logprobs = rollout_outputs.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            if include_action_mask:
                if len(prompt_ids) > MAX_PROMPT_LEN:
                    print(
                        f"Warning: Prompt exceeded {MAX_PROMPT_LEN} tokens ({len(prompt_ids)}) at turn {turn_number}"
                    )
                    termination_reason = "max_prompt_len_exceeded"
                    break

                if turn_number == 0:
                    episode_prompt_ids = prompt_ids
                    prev_full_ids = prompt_ids.copy()
                else:
                    if prev_full_ids is None:
                        prev_full_ids = prompt_ids.copy()
                    elif prompt_ids[: len(prev_full_ids)] != prev_full_ids:
                        prev_full_ids = prompt_ids.copy()
                    else:
                        delta_prompt_ids = prompt_ids[len(prev_full_ids) :]
                        if delta_prompt_ids:
                            episode_completion_ids.extend(delta_prompt_ids)
                            episode_logprobs.extend([0.0] * len(delta_prompt_ids))
                            episode_action_mask.extend([0] * len(delta_prompt_ids))
                        prev_full_ids = prompt_ids.copy()

                if completion_ids:
                    episode_completion_ids.extend(completion_ids)
                    episode_logprobs.extend(logprobs)
                    episode_action_mask.extend([1] * len(completion_ids))
                    if prev_full_ids is not None:
                        prev_full_ids = prev_full_ids + completion_ids
            else:
                prompt_ids_last = prompt_ids
                completion_ids_last = completion_ids
                logprobs_last = logprobs

            messages.append({"role": "assistant", "content": completion_text})

            action_to_send = _parse_action_id(completion_text, legal_action_map)
            parse_failed = not action_to_send
            if parse_failed or action_to_send not in legal_action_map:
                invalid_count += 1
                step_imitation_reward = -RULE_INVALID_ACTION_PENALTY
                action_to_send = _select_fallback_action(legal_action_map)
            elif action_to_send == target_action_id:
                step_imitation_reward = RULE_ACTION_MATCH_REWARD
            else:
                step_imitation_reward = -RULE_ACTION_MISMATCH_PENALTY

            accumulated_imitation_reward += step_imitation_reward

            action_label = legal_action_map.get(action_to_send, "")
            target_bid = target_action_meta.get("selected_bid")
            rule_current_bid_probability = float(target_action_meta.get("current_bid_probability", 0.0))
            rule_selected_bid_support = int(target_action_meta.get("selected_bid_support", 0))
            rule_selected_bid_truth_probability = float(
                target_action_meta.get("selected_bid_truth_probability", 0.0)
            )

            try:
                formatted_observation, step_reward, done, last_step_block = _step_environment(
                    env_endpoint=env_endpoint,
                    episode_id=episode_id,
                    action_to_send=action_to_send,
                    timeout=timeout,
                )
            except Exception as e:
                print(f"Step failed: {e}")
                formatted_observation = ""
                step_reward = -0.01
                done = False
                invalid_count += 1
                accumulated_imitation_reward -= INVALID_ACTION_PENALTY
                last_step_block = {"reward": step_reward, "done": False}

            observation_lower = formatted_observation.lower()
            invalid_or_noop = (
                "invalid" in observation_lower
                or "nothing happens" in observation_lower
                or "nothing happened" in observation_lower
            )
            if invalid_or_noop:
                invalid_count += 1
                accumulated_imitation_reward -= INVALID_ACTION_PENALTY

            if done:
                final_reward = _extract_terminal_reward(last_step_block, formatted_observation)
                termination_reason = "done"
            else:
                messages.append({"role": "user", "content": formatted_observation})

            step_records.append(
                {
                    "turn": turn_number,
                    "assistant_text": trace_logger.clip_text(completion_text) if trace_logger else completion_text,
                    "parsed_action": action_to_send,
                    "action_label": action_label,
                    "observation_before_action": (
                        trace_logger.clip_text(observation_before_action)
                        if trace_logger
                        else observation_before_action
                    ),
                    "observation_after_action": (
                        trace_logger.clip_text(formatted_observation) if trace_logger else formatted_observation
                    ),
                    "step_reward": float(step_reward),
                    "target_action": target_action_id,
                    "target_action_label": target_action_label,
                    "model_action_matches_target": bool(not parse_failed and action_to_send == target_action_id),
                    "step_imitation_reward": float(step_imitation_reward),
                    "rule_current_bid_probability": rule_current_bid_probability,
                    "rule_selected_bid": target_bid,
                    "rule_selected_bid_support": rule_selected_bid_support,
                    "rule_selected_bid_truth_probability": rule_selected_bid_truth_probability,
                    "done": bool(done),
                    "invalid_or_noop": invalid_or_noop,
                    "parse_failed": bool(parse_failed),
                }
            )

            turn_number += 1

        if not done:
            if termination_reason == "unknown":
                termination_reason = "max_turn_reached"
            final_reward = 0.0

        clipped_imitation_reward = _clamp(
            accumulated_imitation_reward,
            -IMITATION_REWARD_CLIP,
            IMITATION_REWARD_CLIP,
        )
        weighted_terminal_reward = TERMINAL_REWARD_WEIGHT * final_reward
        train_reward = clipped_imitation_reward + weighted_terminal_reward

        print(
            f"[ID:{game_id} Done:{int(done)} T:{turn_number:2d} "
            f"Env:{final_reward:+.3f} Imit:{accumulated_imitation_reward:+.3f} "
            f"ClipImit:{clipped_imitation_reward:+.3f} WeightedEnv:{weighted_terminal_reward:+.3f} "
            f"Inv:{invalid_count}"
        )

        if trace_logger and trace_logger.should_log():
            trace_logger.log_episode(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "game_id": game_id,
                    "episode_id": episode_id,
                    "environment": "liars_dice",
                    "status": "completed" if done else "truncated",
                    "termination_reason": termination_reason,
                    "turns": turn_number,
                    "final_reward": float(final_reward),
                    "raw_imitation_reward": float(accumulated_imitation_reward),
                    "clipped_imitation_reward": float(clipped_imitation_reward),
                    "weighted_terminal_reward": float(weighted_terminal_reward),
                    "train_reward": float(train_reward),
                    "invalid_count": invalid_count,
                    "steps": step_records,
                }
            )

        if include_action_mask:
            if len(episode_completion_ids) > MAX_EPISODE_TOKENS:
                episode_completion_ids = episode_completion_ids[:MAX_EPISODE_TOKENS]
                episode_logprobs = episode_logprobs[:MAX_EPISODE_TOKENS]
                episode_action_mask = episode_action_mask[:MAX_EPISODE_TOKENS]

            return index, {
                "prompt_ids": episode_prompt_ids,
                "completion_ids": episode_completion_ids,
                "action_mask": episode_action_mask,
                "logprobs": episode_logprobs,
                "reward": train_reward,
                "final_score": final_reward,
            }

        return index, {
            "prompt_ids": prompt_ids_last,
            "completion_ids": completion_ids_last,
            "logprobs": logprobs_last,
            "reward": train_reward,
            "final_score": final_reward,
        }

    executor = _ROLLOUT_STATE["thread_pool"]
    fallback_builder = _full_prompt_fallback_result if include_action_mask else _last_prompt_fallback_result
    list_results = _execute_parallel_rollouts(
        prompts=prompts,
        executor=executor,
        run_single_prompt=run_single_prompt,
        fallback_builder=fallback_builder,
    )

    curriculum.step(len(prompts))
    _log_batch_statistics(list_results)

    if include_action_mask:
        return {
            "prompt_ids": [r["prompt_ids"] for r in list_results],
            "completion_ids": [r["completion_ids"] for r in list_results],
            "action_mask": [r["action_mask"] for r in list_results],
            "logprobs": [r["logprobs"] for r in list_results],
            "env_rewards": [r["reward"] for r in list_results],
        }

    return {
        "prompt_ids": [r["prompt_ids"] for r in list_results],
        "completion_ids": [r["completion_ids"] for r in list_results],
        "logprobs": [r["logprobs"] for r in list_results],
        "env_rewards": [r["reward"] for r in list_results],
    }


def rollout_last_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    del max_turns  # Curriculum controls effective horizon.
    return _rollout_parallelized_curriculum(prompts=prompts, trainer=trainer, include_action_mask=False)


def rollout_full_prompt_and_completion_parallelized_curriculum(
    prompts: list[str],
    trainer,
    max_turns: int = 30,
) -> dict[str, list]:
    del max_turns  # Curriculum controls effective horizon.
    return _rollout_parallelized_curriculum(prompts=prompts, trainer=trainer, include_action_mask=True)


def rollout_reward_func(completions, **kwargs):
    rewards = kwargs.get("env_rewards") if kwargs else None
    return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)