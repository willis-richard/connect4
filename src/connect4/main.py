"""Hi"""

from src.connect4 import match
from src.connect4 import player

if __name__ == "__main__":
    #player_1 = player.HumanPlayer("player_1", 1)
    player_1 = player.ComputerPlayer("player_1", 1, player.ComputerPlayer.gridsearch, depth=4)
    player_2 = player.ComputerPlayer("player_2", -1, player.ComputerPlayer.gridsearch, depth=4)
    match = match.Match(True, 24, player_1, player_2)
    #match.play()
    match.play_parallel(12)
