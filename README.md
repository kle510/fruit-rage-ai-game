# Fruit Rage AI Game
This project is a Candy Crush-like game playing AI agent that utilizes the “minimax” algorithm with alpha-beta pruning to predict the next best play to win the game with >70% success.

In this zero-sum two player game, each player tries to maximize their share of fruits randomly placed in a grid. The cells in each grid are either empty or filled with one type of fruit. When a player takes a fruit from a cell, the empty slot remaining from the cell is filled the fruits that were on top of it (falling down, i.e. as a result of gravity). The score of each player is determined as the sum of the points acquired from taking fruit for each turn. The game ends when there is no fruit left in the box or when one of the players runs out of the total alloted time. 

Two agents are employed to play against our player: a random agent that selects a random fruit on each turn, and a minimax agent that uses alpha-beta pruning to predict the next best play given a lookahead depth of three. 

