"""Hi"""

from connect4 import game

if __name__ == "__main__":
#    name_o = str(input("Enter player o's name: "))
#    name_x = str(input("Enter player x's name: "))
#    game = game.Connect4(name_o, name_x, True, False)
    game = game.Connect4("one", "two", False, False)
    game.play()
