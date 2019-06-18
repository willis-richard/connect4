# Intro
Using reinforcement learning to train an agent to play Connect4 as well as it can. I use a recursive neural network as a value approximator which also outputs a policy of the likely best move in a position. This is combined with a Monte Carlo Marcov Chain tree search to plan the best move.

This repo is a training project to learn about reinforcement learning. I hope that other people might also be able to learn from my work. I welcome constructive criticism if and reader would like to educate me on 'things I could improve'.

## Acknowledgements
This project is a re-implementation of Deepmind's paper [A general reinforcement learning algorithm that
masters chess, shogi and Go through self-play](https://deepmind.com/documents/260/alphazero_preprint.pdf)

Another resource that has covered this that I found useful was [an Oracle blog on medium](https://medium.com/oracledevs/lessons-from-implementing-alphazero-7e36e9054191). This is what I used as a rough guide to follow in selecting Connect4 as a reasonable game, and a few of the changes

Other useful resources are:
[PascalPons' Github](https://github.com/PascalPons/connect4/tree/a0fcfe9e4eacd6194da8ae138a8e554f381be9e0) for having an efficient computational solution for Connect4. I really wish I had seen this before I had just about finished!
[John's Connect Four Playground](https://tromp.github.io/c4/c4.html)
[The dataset I used for evaluation](http://archive.ics.uci.edu/ml/datasets/connect-4)
[A lecture series by David Silver on Reinforcement Learning](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

Finally I should acknowledge the less glamarous but probably most imporant role that StackOverflow has played.

# Usage
There are three modes that can be used:
$ python main.py -m [mode]
Where mode can be:
game: to play a single game use (typically used to play human vs the AI)
match: to play a match use (typically used to run a AI vs AI to confirm that the training is making it play better in head to head)
training: to run the self-play training loops

I have included a trained network in the base directory called 'example_net.pth'

With the config left as is, my training run produced the following 8ply training loss:

# Notes/Conventions

I use the convention that a position is scored 1 if it is a win for the first player to move in a game of Connect4 ('player_o') and value 0 if the second player will win ('player_x'). Game tree nodes store this absolute position evaluation, but when querying the node for the value, it will return the relative value to the player querying it. i.e. if a position is won for the second player it will have value 0, and if the second player queries the value of this position, a value 1 - 0 = 1 will be returned.

# Training

Here is an example training run:
![alt text](./misc/example_training.pdf)
