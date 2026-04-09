import json
import math
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

SELECTED_GAME = "leduc_poker"
REQUEST_TIMEOUT_SECONDS = 2400
INIT_TIMEOUT_SECONDS = 300
MAX_EPISODE_TOKENS = 16384
MAX_PROMPT_LEN = 16384 - 512

MCTS_CONFIG = {
    "opponent": "mcts",
    "mcts_max_simulations": 50,
    "mcts_num_rollouts": 1,
}

# ---------------------------------------------------------------------------
# Reward-shaping constants
# ---------------------------------------------------------------------------
INVALID_ACTION_PENALTY = 0.10
FOLD_WITH_STRONG_HAND_PENALTY = 0.0
VALUE_BET_BONUS = 0.0
BLUFF_QUALITY_BONUS = 0.0
PASSIVE_WITH_STRONG_PENALTY = 0.0
FOLD_WEAK_HAND_BONUS = 0.0
SHAPING_REWARD_CLIP = 0.10
TERMINAL_REWARD_CLIP = 1.00

# Card strength in Leduc Poker: K > Q > J
CARD_RANK = {"J": 0, "Q": 1, "K": 2}

STRATEGY_TIPS = """
STRATEGY TIPS:
- With a King: Raise for value, especially if it pairs the community card.
- With a Queen: Play cautiously. Call most bets; raise only with a pair.
- With a Jack: Minimize losses. Check/call cheaply; fold against heavy aggression unless you have a pair.
- PAIR (your card matches community card): Always raise aggressively.
- NO PAIR post-flop with low card: Prefer checking/calling. Fold vs large raises.
- Do NOT raise every hand. Selective aggression wins; constant raising is exploitable.
"""

REASONING_TAG_PAIRS = [
    ("think", "think"),
    ("thinking", "thinking"),
    ("reasoning", "reasoning"),
    ("thought", "thought"),
    ("reflection", "reflection"),
]

_ROLLOUT_STATE: dict = {}


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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
    # Leduc Poker observations from the server already contain structured
    # legal-action blocks, similar to Liar's Dice.
    return obs_text or ""


# ---------------------------------------------------------------------------
# Episode trace logger (identical pattern to Liar's Dice)
# ---------------------------------------------------------------------------

class EpisodeTraceLogger:
    """Thread-safe JSONL episode tracer."""

    def __init__(self, trace_dir: str, rank: int):
        self.trace_dir = trace_dir
        self.rank = rank
        self._lock = Lock()
        self.log_path = os.path.join(self.trace_dir, f"leduc_poker_episode_traces_rank{rank}.jsonl")
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


# ---------------------------------------------------------------------------
# Curriculum scheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """Progressive turn-limit curriculum for Leduc Poker.

    Leduc games are short (typically 2-6 player actions), so the curriculum
    ramps from 2 to 8 turns.  The hint probability and MCTS simulation count
    also evolve over training.
    """

    def __init__(
        self,
        initial_max_turn: int = 2,
        final_max_turn: int = 8,
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


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Observation / action parsing
# ---------------------------------------------------------------------------

def _extract_legal_action_map(observation: str) -> dict[str, str]:
    """Parse the ``Legal Actions:`` block into {action_id: label}."""
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


def _extract_state_features(observation: str) -> dict:
    """Extract poker-relevant features from an observation string.

    Expected fields (server-dependent, with regex fallbacks):
    - hole_card: the player's private card (J, Q, or K)
    - community_card: the revealed community card (J, Q, K, or None for round 1)
    - pot_size: total chips in the pot
    - round_number: 1 (pre-flop) or 2 (post-flop)
    - player_chips: player's remaining chips (if available)
    """
    hole_card: str | None = None
    community_card: str | None = None
    pot_size: int = 0
    round_number: int = 1

    # Hole card
    hole_match = re.search(
        r"(?:Your card|Your hole card|Hand|Private card):\s*\[?([JQK])\]?",
        observation,
        flags=re.IGNORECASE,
    )
    if hole_match:
        hole_card = hole_match.group(1).upper()

    # Community card
    comm_match = re.search(
        r"(?:Community card|Public card|Board|Flop):\s*\[?([JQK])\]?",
        observation,
        flags=re.IGNORECASE,
    )
    if comm_match:
        community_card = comm_match.group(1).upper()

    # Pot size
    pot_match = re.search(r"(?:Pot|Pot size):\s*(\d+)", observation, flags=re.IGNORECASE)
    if pot_match:
        pot_size = int(pot_match.group(1))

    # Round
    round_match = re.search(r"(?:Round|Betting round):\s*(\d+)", observation, flags=re.IGNORECASE)
    if round_match:
        round_number = int(round_match.group(1))
    elif community_card is not None:
        round_number = 2

    return {
        "hole_card": hole_card,
        "community_card": community_card,
        "pot_size": pot_size,
        "round_number": round_number,
    }


def _hand_strength(hole_card: str | None, community_card: str | None) -> float:
    """Return a [0, 1] hand-strength estimate.

    Leduc Poker hand ranking:
      - Pair (hole == community):  strongest
      - Among non-pairs, K > Q > J
    Returns a normalised score: 0.0 = weakest, 1.0 = strongest.
    """
    if hole_card is None:
        return 0.5  # unknown

    rank_val = CARD_RANK.get(hole_card, 0)  # J=0, Q=1, K=2

    if community_card is None:
        # Pre-flop: only hole card matters.  K=1.0, Q=0.5, J=0.0
        return rank_val / 2.0

    if hole_card == community_card:
        # Pair — very strong.  K-pair > Q-pair > J-pair
        return 0.8 + 0.1 * rank_val  # 0.8, 0.9, 1.0

    # No pair — rank relative to community.
    return rank_val / 4.0  # J=0.0, Q=0.25, K=0.5


def _classify_action(label: str) -> str:
    """Classify an action label into a semantic category.

    Returns one of: 'fold', 'check', 'call', 'raise', 'unknown'.
    """
    low = (label or "").strip().lower()
    if "fold" in low:
        return "fold"
    if "call" in low or "check" in low:
        return "call"
    if "raise" in low or "bet" in low:
        return "raise"
    return "unknown"


# ---------------------------------------------------------------------------
# Reward-shaping functions
# ---------------------------------------------------------------------------

def _score_action_quality(
    state_features: dict,
    action_label: str,
    legal_action_map: dict[str, str],
) -> float:
    """Compute a per-step shaping reward based on the action taken.

    Shaping components:
    1. **Fold-with-strong penalty** — folding when hand strength >= 0.6
    2. **Value-bet bonus** — raising with strong hand (strength >= 0.7)
    3. **Bluff-quality bonus** — raising in round 1 (pre-flop bluff has merit)
    4. **Passive-with-strong penalty** — checking/calling when you could raise
       with a strong hand post-flop
    """
    hole_card = state_features.get("hole_card")
    community_card = state_features.get("community_card")
    round_number = state_features.get("round_number", 1)
    strength = _hand_strength(hole_card, community_card)
    action_type = _classify_action(action_label)

    reward = 0.0

    # Can the player raise?
    can_raise = any(
        _classify_action(lbl) == "raise"
        for lbl in legal_action_map.values()
    )

    # 1. Fold with strong hand
    if action_type == "fold" and strength >= 0.6:
        reward -= FOLD_WITH_STRONG_HAND_PENALTY

    # 2. Value-bet: raising with a strong hand
    if action_type == "raise" and strength >= 0.7:
        reward += VALUE_BET_BONUS

    # 3. Bluff quality: raising pre-flop with a weak hand is strategically
    #    sound in Leduc (balanced range). Only reward if not obviously strong
    #    to avoid double-counting with value-bet bonus.
    if action_type == "raise" and round_number == 1 and strength <= 0.5:
        reward += BLUFF_QUALITY_BONUS

    # 4. Passive with strong hand post-flop
    if (
        action_type == "call"
        and round_number == 2
        and strength >= 0.7
        and can_raise
    ):
        reward -= PASSIVE_WITH_STRONG_PENALTY

    # 5. Fold-to-save: folding weak hand post-flop is good risk management
    if action_type == "fold" and round_number == 2 and strength < 0.3:
        reward += FOLD_WEAK_HAND_BONUS

    return reward


def _parse_action_id(completion_text: str, legal_action_map: dict[str, str]) -> str:
    """Extract a valid action ID from model output with multiple fallbacks."""
    if not legal_action_map:
        return ""

    cleaned = remove_reasoning_tags(completion_text or "")
    if cleaned.endswith("</s>"):
        cleaned = cleaned[:-5]
    if "Action:" in cleaned:
        cleaned = cleaned.split("Action:")[-1].strip()

    # Try numeric IDs first
    for num in re.findall(r"-?\d+", cleaned):
        if num in legal_action_map:
            return num

    # Try label matching
    normalized = cleaned.strip().lower()
    for action_id, label in legal_action_map.items():
        if normalized == label.strip().lower():
            return action_id

    # Keyword matching: fold/check/call/raise
    for keyword in ("fold", "check", "call", "raise", "bet"):
        if keyword in normalized:
            for action_id, label in legal_action_map.items():
                if keyword in label.lower():
                    return action_id

    return ""


def _select_fallback_action(
    legal_action_map: dict[str, str],
    state_features: dict,
) -> str:
    """Pick a reasonable default action when parsing fails.

    Prefers check > call > fold.  Avoids raising on a fallback.
    """
    # Prefer check
    for action_id, label in legal_action_map.items():
        if _classify_action(label) == "check":
            return action_id
    # Then call
    for action_id, label in legal_action_map.items():
        if _classify_action(label) == "call":
            return action_id
    # Last resort: lowest action ID
    return sorted(legal_action_map.keys(), key=lambda x: int(x))[0]


def _extract_terminal_reward(step_block: dict, observation_text: str) -> float:
    """Pull the terminal reward from the server response."""
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


# ---------------------------------------------------------------------------
# Environment pool / initialisation
# ---------------------------------------------------------------------------

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
    initial_max_turn = int(getattr(trainer.args, "initial_max_turn", 4))
    final_max_turn = int(os.environ.get("LEDUC_POKER_FINAL_MAX_TURN", "8"))
    initial_hint_prob = float(os.environ.get("LEDUC_POKER_INITIAL_HINT_PROB", "0.5"))
    final_hint_prob = float(os.environ.get("LEDUC_POKER_FINAL_HINT_PROB", "0.0"))

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


# ---------------------------------------------------------------------------
# Server interaction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Fallback results
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _get_system_prompt(use_hints: bool) -> str:
    system_prompt = """You are playing leduc_poker.

# Game Rules
LEDUC POKER RULES:

Deck: 2 suits x 3 ranks. For 2 players: 6 cards (J, Q, K in two suits).

Setup: Each player starts with 100 chips, pays 1 ante. Two rounds of betting.

Round 1: Each player receives one private card.
Actions: Fold (lose ante), Call/Check (match current bet or pass), Raise (add 2 chips to bet).
Maximum 2 raises per round.

Round 2: One public card is revealed. Same actions, but Raise adds 4 chips.

Winning: Player with best hand wins pot (or last remaining if others fold).
Hand ranking (high to low): Pair (private + public match) > High card value (K > Q > J).

# Output Format
You must respond with ONLY the action ID (a single number).
Do NOT include descriptions or explanations.
Examples:
- For action "0 -> Fold": respond "0"
- For action "1 -> Call": respond "1"
- For action "2 -> Raise": respond "2"

CRITICAL: Your entire response must be a single number. No words, no punctuation, no explanation.
"""
    if use_hints:
        system_prompt += "\n" + STRATEGY_TIPS
    return system_prompt


# ---------------------------------------------------------------------------
# Main rollout
# ---------------------------------------------------------------------------

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
        accumulated_shaping_reward = 0.0
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
            state_features = _extract_state_features(observation_before_action)

            if not legal_action_map:
                accumulated_shaping_reward -= INVALID_ACTION_PENALTY
                termination_reason = "no_legal_actions"
                break

            with _ROLLOUT_STATE["generation_semaphore"]:
                rollout_outputs = generate_rollout_completions(
                    trainer, prompts=[messages], as_chat=True
                )[0]

            prompt_ids = rollout_outputs.get("prompt_ids", [])
            completion_ids = rollout_outputs.get("completion_ids", [])
            logprobs = rollout_outputs.get("logprobs", [])
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

            # --- Action masking bookkeeping ---
            if include_action_mask:
                if len(prompt_ids) > MAX_PROMPT_LEN:
                    print(
                        f"Warning: Prompt exceeded {MAX_PROMPT_LEN} tokens "
                        f"({len(prompt_ids)}) at turn {turn_number}"
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

            # --- Parse action ---
            action_to_send = _parse_action_id(completion_text, legal_action_map)
            parse_failed = not action_to_send
            if parse_failed or action_to_send not in legal_action_map:
                invalid_count += 1
                accumulated_shaping_reward -= INVALID_ACTION_PENALTY
                action_to_send = _select_fallback_action(legal_action_map, state_features)

            action_label = legal_action_map.get(action_to_send, "")

            # --- Reward shaping ---
            action_shaping = _score_action_quality(
                state_features, action_label, legal_action_map
            )
            accumulated_shaping_reward += action_shaping

            # --- Step environment ---
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
                accumulated_shaping_reward -= INVALID_ACTION_PENALTY
                last_step_block = {"reward": step_reward, "done": False}

            observation_lower = formatted_observation.lower()
            invalid_or_noop = (
                "invalid" in observation_lower
                or "nothing happens" in observation_lower
                or "nothing happened" in observation_lower
            )
            if invalid_or_noop:
                invalid_count += 1
                accumulated_shaping_reward -= INVALID_ACTION_PENALTY

            if done:
                # Use step_reward directly (like gin_rummy) — _extract_terminal_reward
                # can return 0 when info.cumulative_reward is 0, masking the real outcome.
                final_reward = _clamp(step_reward, -TERMINAL_REWARD_CLIP, TERMINAL_REWARD_CLIP)
                termination_reason = "done"
            else:
                messages.append({"role": "user", "content": formatted_observation})

            step_records.append(
                {
                    "turn": turn_number,
                    "assistant_text": (
                        trace_logger.clip_text(completion_text) if trace_logger else completion_text
                    ),
                    "parsed_action": action_to_send,
                    "action_label": action_label,
                    "action_type": _classify_action(action_label),
                    "hand_strength": _hand_strength(
                        state_features.get("hole_card"),
                        state_features.get("community_card"),
                    ),
                    "action_shaping": float(action_shaping),
                    "observation_before_action": (
                        trace_logger.clip_text(observation_before_action)
                        if trace_logger
                        else observation_before_action
                    ),
                    "observation_after_action": (
                        trace_logger.clip_text(formatted_observation)
                        if trace_logger
                        else formatted_observation
                    ),
                    "step_reward": float(step_reward),
                    "done": bool(done),
                    "invalid_or_noop": invalid_or_noop,
                    "parse_failed": bool(parse_failed),
                }
            )

            turn_number += 1

        # --- Episode reward ---
        if not done:
            if termination_reason == "unknown":
                termination_reason = "max_turn_reached"
            final_reward = 0.0

        clipped_shaping = _clamp(
            accumulated_shaping_reward, -SHAPING_REWARD_CLIP, SHAPING_REWARD_CLIP
        )
        train_reward = final_reward + clipped_shaping

        print(
            f"[ID:{game_id} Done:{int(done)} T:{turn_number:2d} "
            f"Env:{final_reward:+.3f} Shape:{accumulated_shaping_reward:+.3f} "
            f"ClipShape:{clipped_shaping:+.3f} Inv:{invalid_count}"
        )

        if trace_logger and trace_logger.should_log():
            trace_logger.log_episode(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "game_id": game_id,
                    "episode_id": episode_id,
                    "environment": "leduc_poker",
                    "status": "completed" if done else "truncated",
                    "termination_reason": termination_reason,
                    "turns": turn_number,
                    "final_reward": float(final_reward),
                    "raw_shaping_reward": float(accumulated_shaping_reward),
                    "clipped_shaping_reward": float(clipped_shaping),
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


# ---------------------------------------------------------------------------
# Public API (matches the interface expected by train_grpo_env.py)
# ---------------------------------------------------------------------------

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
