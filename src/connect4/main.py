"""Hi"""

from src.connect4 import match
from src.connect4 import player

if __name__ == "__main__":
    #player_1 = player.HumanPlayer("player_1", 1)
    player_1 = player.ComputerPlayer("player_1", 1, 4)
    player_2 = player.ComputerPlayer("player_2", -1, 4)
    match = match.Match(6, player_1, player_2)
    match.play()