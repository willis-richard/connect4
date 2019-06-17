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
There are three modes that 
