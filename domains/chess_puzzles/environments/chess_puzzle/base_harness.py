import re

SYSTEM_PROMPT = """You solve chess tactics by choosing one legal move at a time.
Use only the board position and legal moves shown in the prompt. Do not claim
hidden engine analysis or external knowledge. Return exactly one move."""

USER_PROMPT = """Solve the current chess puzzle position.

{observation}

Return exactly one move as <move>uci</move>, for example <move>e2e4</move>.
Do not include explanation."""

FORMAT_RETRY_BUDGET = 1
ILLEGAL_MOVE_RETRY_BUDGET = 1
MAX_TURNS = 9


def _legal_moves_from_observation(observation: str) -> set[str]:
    moves = set()
    for match in re.finditer(r"\[\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*\]", observation, re.I):
        moves.add(match.group(1).lower())
    return moves


def _uci_candidates(text: str) -> list[str]:
    candidates = []
    for pattern in [
        r"<move>\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*</move>",
        r"\[\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*\]",
        r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b",
    ]:
        for match in re.finditer(pattern, text, re.I):
            candidates.append(match.group(1).lower())
        if candidates:
            return candidates
    return []


def parse_action(response: str) -> str:
    candidates = _uci_candidates(str(response))
    unique = []
    for move in candidates:
        if move not in unique:
            unique.append(move)
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        return "__ambiguous_multiple_moves__"
    return "__no_move__"


def format_observation(observation: str, **kwargs) -> str:
    return str(observation).strip()


def propose_action(board: str) -> str:
    return ""


def is_legal_action(board: str, action: str) -> bool:
    if not isinstance(action, str):
        return False
    action = action.strip().lower()
    if not re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", action):
        return False
    return action in _legal_moves_from_observation(str(board))
