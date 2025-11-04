from dataclasses import dataclass
from typing import List, Tuple, Optional
import random, math

@dataclass
class MancalaState:
    pits: List[int]
    player_to_move: int
    last_move_was_capture: bool = False
    game_over: bool = False

    @staticmethod
    def standard_start(stones_per_pit: int = 4) -> "MancalaState":
        pits = [stones_per_pit]*6 + [0] + [stones_per_pit]*6 + [0]
        return MancalaState(pits=pits, player_to_move=0)

    def side_range(self, player: int):
        return range(0, 6) if player == 0 else range(7, 13)

    def mancala_index(self, player: int) -> int:
        return 6 if player == 0 else 13

    def opponent_mancala_index(self, player: int) -> int:
        return 13 if player == 0 else 6

    def legal_moves(self, player: Optional[int] = None):
        if player is None: player = self.player_to_move
        r = self.side_range(player)
        return [i for i in r if self.pits[i] > 0]

    def clone(self) -> "MancalaState":
        return MancalaState(self.pits[:], self.player_to_move, self.last_move_was_capture, self.game_over)

def opposite_index(i: int) -> int:
    return 12 - i

def apply_move(state: MancalaState, move_idx: int, continuation_rule: bool=False) -> MancalaState:
    s = state.clone()
    player = s.player_to_move
    assert move_idx in s.legal_moves(player), "Illegal move"
    stones = s.pits[move_idx]
    s.pits[move_idx] = 0

    idx = move_idx
    while stones > 0:
        idx = (idx + 1) % 14
        if idx == s.opponent_mancala_index(player):
            continue
        s.pits[idx] += 1
        stones -= 1

    s.last_move_was_capture = False
    if idx in s.side_range(player) and s.pits[idx] == 1:
        opp = opposite_index(idx)
        captured = s.pits[opp]
        if captured > 0:
            s.pits[opp] = 0
            s.pits[idx] = 0
            s.pits[s.mancala_index(player)] += captured + 1
            s.last_move_was_capture = True

    def side_empty(p: int) -> bool:
        return sum(s.pits[i] for i in s.side_range(p)) == 0

    if side_empty(0) or side_empty(1):
        for p in (0,1):
            r = s.side_range(p)
            stash = sum(s.pits[i] for i in r)
            for i in r: s.pits[i] = 0
            s.pits[s.mancala_index(p)] += stash
        s.game_over = True

    if not s.game_over:
        if continuation_rule and idx == s.mancala_index(player):
            s.player_to_move = player
        else:
            s.player_to_move = 1 - player

    return s

def utility(state: MancalaState, max_player: int) -> int:
    return state.pits[state.mancala_index(max_player)] - state.pits[state.mancala_index(1 - max_player)]

def play_game(p0_policy, p1_policy, continuation_rule: bool=False, seed: Optional[int]=None, max_moves: int=500):
    if seed is not None:
        random.seed(seed)
    s = MancalaState.standard_start()
    moves = 0
    while not s.game_over and moves < max_moves:
        player = s.player_to_move
        policy = p0_policy if player == 0 else p1_policy
        legal = s.legal_moves()
        if not legal:
            s.player_to_move = 1 - player
            continue
        m = policy(s, legal, player)
        s = apply_move(s, m, continuation_rule=continuation_rule)
        moves += 1
    return s, moves

def random_policy(state: MancalaState, legal_moves, player: int) -> int:
    return random.choice(legal_moves)

def minimax_policy(depth: int):
    def policy(state: MancalaState, legal_moves, player: int) -> int:
        best_score = -math.inf
        best_move = legal_moves[0]
        for m in legal_moves:
            child = apply_move(state, m, continuation_rule=False)
            score = _min_value(child, depth-1, max_player=player)
            if score > best_score:
                best_score = score
                best_move = m
        return best_move
    return policy

def _terminal_or_depth(state: MancalaState, depth: int) -> bool:
    return state.game_over or depth == 0

def _max_value(state: MancalaState, depth: int, max_player: int) -> int:
    if _terminal_or_depth(state, depth):
        return utility(state, max_player)
    legal = state.legal_moves()
    if not legal:
        s2 = state.clone()
        s2.player_to_move = 1 - state.player_to_move
        return _min_value(s2, depth, max_player)
    v = -math.inf
    for m in legal:
        child = apply_move(state, m, continuation_rule=False)
        v = max(v, _min_value(child, depth-1, max_player))
    return v

def _min_value(state: MancalaState, depth: int, max_player: int) -> int:
    if _terminal_or_depth(state, depth):
        return utility(state, max_player)
    legal = state.legal_moves()
    if not legal:
        s2 = state.clone()
        s2.player_to_move = 1 - state.player_to_move
        return _max_value(s2, depth, max_player)
    v = math.inf
    for m in legal:
        child = apply_move(state, m, continuation_rule=False)
        v = min(v, _max_value(child, depth-1, max_player))
    return v

def alphabeta_policy(depth: int):
    def policy(state: MancalaState, legal_moves, player: int) -> int:
        alpha, beta = -math.inf, math.inf
        best_score = -math.inf
        best_move = legal_moves[0]
        for m in legal_moves:
            child = apply_move(state, m, continuation_rule=False)
            score = _ab_min(child, depth-1, player, alpha, beta)
            if score > best_score:
                best_score = score
                best_move = m
            alpha = max(alpha, best_score)
        return best_move
    return policy

def _ab_max(state: MancalaState, depth: int, max_player: int, alpha: float, beta: float) -> int:
    if _terminal_or_depth(state, depth):
        return utility(state, max_player)
    legal = state.legal_moves()
    if not legal:
        s2 = state.clone()
        s2.player_to_move = 1 - state.player_to_move
        return _ab_min(s2, depth, max_player, alpha, beta)
    v = -math.inf
    for m in legal:
        child = apply_move(state, m, continuation_rule=False)
        v = max(v, _ab_min(child, depth-1, max_player, alpha, beta))
        if v >= beta:
            return v
        alpha = max(alpha, v)
    return v

def _ab_min(state: MancalaState, depth: int, max_player: int, alpha: float, beta: float) -> int:
    if _terminal_or_depth(state, depth):
        return utility(state, max_player)
    legal = state.legal_moves()
    if not legal:
        s2 = state.clone()
        s2.player_to_move = 1 - state.player_to_move
        return _ab_max(s2, depth, max_player, alpha, beta)
    v = math.inf
    for m in legal:
        child = apply_move(state, m, continuation_rule=False)
        v = min(v, _ab_max(child, depth-1, max_player, alpha, beta))
        if v <= alpha:
            return v
        beta = min(beta, v)
    return v
