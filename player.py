import argparse
import numpy as np

from othello import Env, Move
from players.base import BasePlayer

# 盤面のスコア配列
scores = np.array(
    [
        [100, -20, 10, 5, 5, 10, -20, 100],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [10, -2, 15, 3, 3, 15, -2, 10],
        [5, -2, 3, 0, 0, 3, -2, 5],
        [5, -2, 3, 0, 0, 3, -2, 5],
        [10, -2, 15, 3, 3, 15, -2, 10],
        [-20, -50, -2, -2, -2, -2, -50, -20],
        [100, -20, 10, 5, 5, 10, -20, 100],
    ],
    dtype="int32",
)

def alphabeta(env, move, depth, alpha, beta, max_depth, is_maximizing_player):
    env.update(move)

    if depth >= max_depth:
        score = move.player * np.sum(env.board * scores)
    else:
        if is_maximizing_player:  # Maximize player
            best_score = -np.inf
            for next_move in env.legal_moves():
                score = alphabeta(env, next_move, depth + 1, alpha, beta, max_depth, False)
                best_score = max(best_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
        else:  # Minimize player
            best_score = np.inf
            for next_move in env.legal_moves():
                score = alphabeta(env, next_move, depth + 1, alpha, beta, max_depth, True)
                best_score = min(best_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break

        score = best_score

    env.undo()
    return score


class MyPlayer(BasePlayer):
    BASE_MAX_DEPTH = 4  # 基本の最大深さ
    DEEPENING_THRESHOLD = 10  # 深く読む閾値（空きマスの数）

    def __init__(self):
        super(MyPlayer, self).__init__()

    def reset(self) -> None:
        pass

    # 空きマスに基づいて読みの深さを調整
    def adjust_depth(self, env):
        empty_cells = np.sum(env.board == 0)
        if empty_cells <= self.DEEPENING_THRESHOLD:
            return self.BASE_MAX_DEPTH + (self.DEEPENING_THRESHOLD - empty_cells)//2
        return self.BASE_MAX_DEPTH
        
    def play(self, env: Env) -> Move:
        moves = env.legal_moves()
        best_move = moves[0]
        best_score = -np.inf
        alpha = -np.inf
        beta = np.inf
        current_depth = self.adjust_depth(env)
        for move in moves:
            score = alphabeta(env, move, 0, alpha, beta, current_depth, False)
            if best_score < score:
                best_move = move
                best_score = score

        return best_move
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Othello player")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    player = MyPlayer()
    player.run(args.verbose)
