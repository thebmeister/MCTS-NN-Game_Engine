"""
connect4state.py
~~~~~~~~~~~~~~~~

This is a class defining the game state for Connect Four.
To be used in conjunction with the MCTS and Neural Network code
to train a Connect Four engine.
"""

class Connect4State(object):
	""" A state of the game, i.e. the game 'board' - 
		I'm going to refer to it as the board from now on.
		The board is arranged as follows:
		35 36 37 38 39 40 41
		28 29 30 31 32 33 34
		21 22 23 24 25 26 27
		14 15 16 17 18 19 20 
		7  8  9  10 11 12 13
		0  1  2  3  4  5  6
		where 0 = empty, 1 = player 1, 2 = player 2
	"""
	def __init__(self):
		self.playerJustMoved = 2 # At the root pretend the player just moved is p2 - p1 has the first move
		self.board = [] # 0 = empty, 1 = player 1, 2 = player 2
		for i in range(6):
			self.board.append([0]*7)

	def Clone(self):
		""" Create a deep clone of this game state.
		"""
		st = Connect4State()
		st.playerJustMoved = self.playerJustMoved
		st.board = self.board
		return st

	def DoMove(self, move):
		""" Update a state by carrying out the given move.
			Must update playerJustMoved.
		"""
		self.playerJustMoved = 3 - self.playerJustMoved

	def UndoMove(self, lastmove):
		""" Update a state by undoing the given move.
			Must update playerJustMoved.
		"""
		self.playerJustMoved = 3 - self.playerJustMoved

	def GetMoves(self):
		""" Get all possible moves from this state.
		"""

	def GetResult(self, playerjm):
		""" Get the game result from the viewpoint of playerjm. 
		"""

	def __repr__(self):
		""" Don't need this - but good style.
		"""
		pass
