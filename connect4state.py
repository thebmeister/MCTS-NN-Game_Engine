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
		self.heights = [0]*7

	def Clone(self):
		""" Create a deep clone of this game state.
		"""
		st = Connect4State()
		st.playerJustMoved = self.playerJustMoved
		st.board = self.board
		st.heights = self.heights
		return st

	def DoMove(self, move):
		""" Update a state by carrying out the given move.
			Must update playerJustMoved.
		"""
		self.playerJustMoved = 3 - self.playerJustMoved
		assert self.heights[move] < 6
		self.board[self.heights[move]][move] = self.playerJustMoved
		self.heights[move] += 1
		
	def UndoMove(self, lastmove):
		""" Update a state by undoing the given move.
			Must update playerJustMoved.
		"""
		self.heights[lastmove] -= 1
		self.board[self.heights[lastmove]][lastmove] = 0
		self.playerJustMoved = 3 - self.playerJustMoved

	def GetMoves(self):
		""" Get all possible moves from this state.
			in pseudocode:
			First check if game is over.
			if yes: no moves left
			if no: return all columns that haven't been filled to the top
		"""
		winlocations  = () # update with all 69 possible win locations as a tuple of tuple of tuples
		winchecks = []
		for location in winlocations:
			wc = []
			for height, column in location:
				wc.append(state.board[height][column])
			winchecks.append(wc)
		for w, x, y, z in winchecks:  
			if w = x = y = z != 0:
				return []
		moves = [i for i in range(6) if self.heights[i] < 6]
		return moves

	def GetResult(self, playerjm):
		""" Get the game result from the viewpoint of playerjm. 
		"""
		winlocations  = () # update with all 69 possible win locations as a tuple of tuple of tuples
		winchecks = []
		for location in winlocations:
			wc = []
			for height, column in location:
				wc.append(state.board[height][column])
			winchecks.append(wc)
		for w, x, y, z in winchecks:  
			if w = x = y = z != 0:
				if self.playerJustMoved == playerjm:
					return 1
				else:
					return 0
		if self.GetMoves() == []: return 0.5 # draw
		assert False # shouldn't be possible to get here
				

	def __repr__(self):
		""" Don't need this - but good style.
		"""
		pass
