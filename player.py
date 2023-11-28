import random

from othello import Move
from players.base import BasePlayer


class MyPlayer(BasePlayer):
    def __init__(self):
        super(MyPlayer, self).__init__()

    def reset(self) -> None:
        # Nothing to do
        pass

    def play(self, env) -> Move:
        moves = env.legal_moves()
        if len(moves) != 0:
            move = random.choice(moves)
        else:
            move = Move.Pass(env.player)

        return move


if __name__ == "__main__":
    player = MyPlayer()
    player.run()
