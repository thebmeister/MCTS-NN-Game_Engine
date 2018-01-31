"""
connect4engine_allcode.py
~~~~~~~~~~~~~~~~~~~~~~~~~
Another messy conglomeration of code.
This time we're training a connect 4 engine, which should be more
fun to play against hopefully. Also a better guage of the effectiveness
of the method for training the engine.
This time, I at least deleted most of the code that wasn't being used.
"""
"""
network.py
~~~~~~~~~~
This is original comment from Michael Nielsen's original code:
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        a =  np.transpose(np.array(a, ndmin = 2))
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w,a) + b
            a = sigmoid(z)
        return a

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation =  np.transpose(np.array(x, ndmin = 2))
        activations = [activation] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = activations[-1] - y
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, np.transpose(activations[-2]))
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(np.transpose(self.weights[-l+1]), delta)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, np.transpose(activations[-l-1]))
        return (nabla_b, nabla_w)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

"""
MCTS.py
~~~~~~~
Below is the comment from the original code which I, Brendan Milway, have modified for my purposes:
This is a very simple implementation of the UCT Monte Carlo Tree Search algorithm in Python 2.7.
The function UCT(rootstate, itermax, verbose = False) is towards the bottom of the code.
It aims to have the clearest and simplest possible code, and for the sake of clarity, the code
is orders of magnitude less efficient than it could be made, particularly by using a 
state.GetRandomMove() or state.DoRandomRollout() function.
Example GameState classes for Nim, OXO and Othello are included to give some idea of how you
can write your own GameState use UCT in your 2-player game. Change the game to be played in 
the UCTPlayGame() function at the bottom of the code.
 
Written by Peter Cowling, Ed Powley, Daniel Whitehouse (University of York, UK) September 2012.
 
Licence is granted to freely use and distribute for any sensible/legal purpose so long as this comment
remains in any distributed code.
For more information about Monte Carlo Tree Search check out our web site at www.mcts.ai
""" 


class Connect4State(object):
	""" A state of the game, i.e. the game 'board' - 
		I'm going to refer to it as the board from now on.
		The board is arranged as follows with (height, column) indices:
		(5,0)  (5,1)  (5,2)  (5,3)  (5,4)  (5,5)  (5,6)
		(4,0)  (4,1)  (4,2)  (4,3)  (4,4)  (4,5)  (4,6)
		(3,0)  (3,1)  (3,2)  (3,3)  (3,4)  (3,5)  (3,6)
		(2,0)  (2,1)  (2,2)  (2,3)  (2,4)  (2,5)  (2,6)
		(1,0)  (1,1)  (1,2)  (1,3)  (1,4)  (1,5)  (1,6)
		(0,0)  (0,1)  (0,2)  (0,3)  (0,4)  (0,5)  (0,6)
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
			for h, c in location:
				wc.append(self.board[h][c])
			winchecks.append(wc)
		for w, x, y, z in winchecks:  
			if w == x == y == z != 0:
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
			for h, c in location:
				wc.append(self.board[h][c])
			winchecks.append(wc)
		for w, x, y, z in winchecks:  
			if w == x == y == z != 0:
				if self.playerJustMoved == playerjm:
					return 1
				else:
					return 0
		if self.GetMoves() == []: return 0.5 # draw
		assert False # shouldn't be possible to get here
				

	def __repr__(self):
		""" Don't need this - but good style.
		"""
		places = ((5,0)  (5,1)  (5,2)  (5,3)  (5,4)  (5,5)  (5,6)\
				  (4,0)  (4,1)  (4,2)  (4,3)  (4,4)  (4,5)  (4,6)\
				  (3,0)  (3,1)  (3,2)  (3,3)  (3,4)  (3,5)  (3,6)\
				  (2,0)  (2,1)  (2,2)  (2,3)  (2,4)  (2,5)  (2,6)\
				  (1,0)  (1,1)  (1,2)  (1,3)  (1,4)  (1,5)  (1,6)\
				  (0,0)  (0,1)  (0,2)  (0,3)  (0,4)  (0,5)  (0,6))
		s = " "
		for h, c in places:
			s += "_12"[self.board[h][c]] + " "
			if c == 6: s += "\n "
		return s
		
class Node:
    """ A node in the game tree. Note wins is always from the viewpoint of playerJustMoved.
        Crashes if state not specified.
    """
    def __init__(self, move = None, parent = None, state = None):
        self.move = move # the move that got us to this node - "None" for the root node
        self.parentNode = parent # "None" for the root node
        self.childNodes = []
        self.wins = 0
        self.visits = 0
        self.untriedMoves = state.GetMoves() # future child nodes
        self.playerJustMoved = state.playerJustMoved # the only part of the state that the Node needs later
        
    def UCTSelectChild(self):
        """ Use the UCB1 formula to select a child node. Often a constant UCTK is applied so we have
            lambda c: c.wins/c.visits + UCTK * sqrt(2*log(self.visits)/c.visits to vary the amount of
            exploration versus exploitation.
        """
        s = sorted(self.childNodes, key = lambda c: c.wins/c.visits + np.sqrt(2*np.log(self.visits)/c.visits))[-1]
        return s
    
    def AddChild(self, m, s):
        """ Remove m from untriedMoves and add a new child node for this move.
            Return the added child node
        """
        n = Node(move = m, parent = self, state = s)
        self.untriedMoves.remove(m)
        self.childNodes.append(n)
        return n
    
    def Update(self, result):
        """ Update this node - one additional visit and result additional wins. result must be from the viewpoint of playerJustmoved.
        """
        self.visits += 1
        self.wins += result

    def __repr__(self):
        return "[M:" + str(self.move) + " W/V:" + str(self.wins) + "/" + str(self.visits) + " U:" + str(self.untriedMoves) + "]"

    def TreeToString(self, indent):
        s = self.IndentString(indent) + str(self)
        for c in self.childNodes:
             s += c.TreeToString(indent+1)
        return s

    def IndentString(self,indent):
        s = "\n"
        for i in range (1,indent+1):
            s += "| "
        return s

    def ChildrenToString(self):
        s = ""
        for c in self.childNodes:
             s += str(c) + "\n"
        return s
	
"""
MUCT_training.py
~~~~~~~~~~~~~~~~
Here is where we train the engine. We start by initializing the neural network 
with random weights. After each practice session we update the network based on 
the results of the modified-UCT player vs itself. As the network improves, the 
modified-UCT player improves, and the results from his games against himself give 
better training data. This in turn improves the performance of the neural network.
The resulting feedback loop rapidly improves the skill of the eninge until,
eventually, it reaches an expert level.
"""
# import numpy as np

def softmax(values):
    tot = sum(np.exp(values))
    return [i/tot for i in np.exp(values)]

def MUCT(rootstate, itermax, verbose = False):
    """ Conduct a modified UCT search for itermax iterations starting from rootstate.
        Rollouts moves are chosen based on the neural network.
        Return the best move from the rootstate.
        Assumes 2 alternating players (player 1 starts), with game results in the range [0.0, 1.0]."""

    rootnode = Node(state = rootstate)

    for i in range(itermax):
        node = rootnode
        state = rootstate.Clone()

        # Select
        while node.untriedMoves == [] and node.childNodes != []: # node is fully expanded and non-terminal
            node = node.UCTSelectChild()
            state.DoMove(node.move)

        # Expand
        if node.untriedMoves != []: # if we can expand (i.e. state/node is non-terminal)
            m = random.choice(node.untriedMoves) 
            state.DoMove(m)
            node = node.AddChild(m,state) # add child and descend tree
			
        # Rollout - this can often be made orders of magnitude quicker using a state.GetRandomMove() function
        while state.GetMoves() != []: # while state is non-terminal
            boardevals = []
            for move in state.GetMoves():
                state.DoMove(move)
                newboard = state.board
                boardeval = EvalNetwork.feedforward(newboard)[0, 0]
                boardevals.append(boardeval)
                state.UndoMove(move)
            weights = softmax(boardevals)
            assert len(weights) == len(state.GetMoves())
            # assert sum(weights) == 1 # this should be true though...
            rollmov = np.random.choice(state.GetMoves(), p = weights)
            state.DoMove(rollmov)

        # Backpropagate
        while node != None : # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(node.playerJustMoved)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose): print(rootnode.TreeToString(0))
    # else: print(rootnode.ChildrenToString())

    return sorted(rootnode.childNodes, key = lambda c: c.visits)[-1].move # return the move that was most visited

def MUTCPlayGame():
    """ Play a sample game between to players using a UTC method modified by the Neural Network evaluation to select
        rollouts that are likely instead of just random. Backpropagates to returns list of tuples, with the state.board
    """
    # state = OthelloState(4) # uncomment to play Othello on a square board of the given size
    # state = OXOState() # uncomment to play OXO
    state = Connect4State()
    # state = NimState(15) # uncomment to play Nim with the given number of starting chips
    boards = []
    while (state.GetMoves() != []):
        m = MUCT(rootstate = state, itermax = 12, verbose = False) # play with values for itermax and verbose = True
        state.DoMove(m)
        simpleboard = [piece for layer in state.board for piece in layer]
        boards.append(simpleboard)
    if state.GetResult(state.playerJustMoved) == 1:
        result = 1
        results = []
        for board in boards:
            results.append(result)
            result = 1 - result
        results.reverse()
        data = [(x,y) for x, y in zip(boards, results)]
        return data
    else:
        result = 0.5
        results = []
        for board in boards:
            results.append(result)
        data = [(x,y) for x, y in zip(boards, results)]
        return data

def Practice_Session(minibatch_size, eta):
    training_data = []
    for i in range(minibatch_size):
        d = MUTCPlayGame()
        training_data += d
    EvalNetwork.update_mini_batch(training_data, eta)

def Train(practice_sessions, minibatch_size, eta):
	for s in range(practice_sessions):
		Practice_Session(minibatch_size, eta)
		print("practice session {0} completed\n".format(s))
	print("training complete\n")

def GetHumanMove(state):
    """ Ask human player for move and determine if it's legal.
		If not, ask again - if so, return that move
    """
    m = eval(input("These are the legal moves:\n" + str(state.GetMoves()) + "\nWhat's your move?\n"))
    legal = 0
    for legal_move in state.GetMoves():
        if m == legal_move:
            legal += 1
    while legal == 0:
        m = eval(input("Error: Invalid move.\nThese are the legal moves:\n" + str(state.GetMoves()) + "\n"))
        for legal_move in state.GetMoves():
            if m == legal_move:
                legal += 1
    return m

def MUCTPlayHuman():
    """ Play a sample game between MUCT player and human. Computer plays first.
    Number of MUCT iterations is adjustable
    """
    playagain = "y"
    while playagain == "y":
        # state = OthelloState(4) # uncomment to play Othello on a square board of the given size
        # state = OXOState() # uncomment to play OXO
        state = Connect4State()
        # state = NimState(15) # uncomment to play Nim with the given number of starting chips
        iterations = eval(input("How many UTC iterations does the Computer opponent get?\n(The more iterations the better it plays, but any \nnumber above 10000 will cause it to run slowly)\n"))
        while isinstance(iterations, int) != True or iterations < 0:
            iterations = eval(input("Error: Please input a positive integer.\n"))
        humanplayer = eval(input("Would you like to be player 1 or 2?\n"))
        while humanplayer != 1 and humanplayer != 2:
            humanplayer = eval(input("Error: Please input either 1 or 2.\n"))
        while (state.GetMoves() != []):
            print(repr(state))
            if state.playerJustMoved == humanplayer:
                m = MUCT(rootstate = state, itermax = iterations, verbose = False)
            else:
                m = GetHumanMove(state)
            state.DoMove(m)
        if state.GetResult(state.playerJustMoved) == humanplayer:
            print(repr(state) + "\nYou win!!\n")
        elif state.GetResult(state.playerJustMoved) == (3 - humanplayer):
            print(repr(state) + "\nYou lose.\n")
        else:
            print(repr(state) + "\nIt's a draw!\n")
        playagain = input("Play Again? (y or n)\n")
        while playagain != "y" and playagain != "n":
            playagain = input("Error: Please input either y or n.\n")
    print("Goodbye.")
							  
if __name__ == "__main__":
	""" Play as human v the MUCT player
	"""
	# hiddenlayerneurons = eval(input("How many hidden layer neurons for this model?\n"))
	# while isinstance(hiddenlayerneurons, int) != True or hiddenlayerneurons < 0:
	#         hiddenlayerneurons = eval(input("Error: Please input a positive integer.\n"))
	networksizes = [42,50,1]# [42, hiddenlayerneurons, 1]
	eta = 0.1 # eval(input("What learning rate would you like to use?\n"))
	# while isinstance(eta, float) != True or eta < 0 or eta > 0.5:
	#         eta = eval(input("Error: Please input a positive number less than 0.5.\n"))
	minibatchsize = 10# eval(input("How many games per practice session?\n"))
	# while isinstance(minibatchsize, int) != True or minibatchsize < 0:
	#         minibatchsize = eval(input("Error: Please input a positive integer.\n"))
	numsessions = 11# eval(input("How many practice sessions to complete training?\n"))
	# while isinstance(numsessions, int) != True or numsessions < 0:
	#         numsessions = eval(input("Error: Please input a positive integer.\n"))
	EvalNetwork = Network(networksizes)
	Train(numsessions, minibatchsize, eta)
	MUCTPlayHuman()

"""
IDEAS FOR FUTURE IMPROVEMENTS
~~~~~~~~~~~
- Would like to include method for storing the trained network in a file.
- Would like to create interface for playing the games--probably importing a different file.
"""
