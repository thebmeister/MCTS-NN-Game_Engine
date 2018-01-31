"""
OXOengine_allcode
~~~~~~~~~~~~~~~~
This messy conglomeration is the first working code I made that successfully 
incorporates neural network into the MCTS, and trains the Neural Network on the games it plays against itself.
The code should be split into distinct files, but I was having trouble with importing the files so I just
copy-pasted everything into this one.
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
    """
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):"""
    """
        Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
    """
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)
    """
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
            print(w.shape)
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        print(zs)
        delta = activations[-1] - y
        print(delta.shape)
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
            print(delta.shape)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, np.transpose(activations[-l-1]))
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

"""
MCTS.py
~~~~~~~
Below is the comment from the original code which I, Brendan Milway, have added bits and pieces to:
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


class GameState:
    """ A state of the game, i.e. the game board. These are the only functions which are
        absolutely necessary to implement UCT in any 2-player complete information deterministic 
        zero-sum game, although they can be enhanced and made quicker, for example by using a 
        GetRandomMove() function to generate a random move during rollout.
        By convention the players are numbered 1 and 2.
    """
    def __init__(self):
            self.playerJustMoved = 2 # At the root pretend the player just moved is player 2 - player 1 has the first move
        
    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = GameState()
        st.playerJustMoved = self.playerJustMoved
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
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


class NimState:
    """ A state of the game Nim. In Nim, players alternately take 1,2 or 3 chips with the 
        winner being the player to take the last chip. 
        In Nim any initial state of the form 4n+k for k = 1,2,3 is a win for player 1
        (by choosing k) chips.
        Any initial state of the form 4n is a win for player 2.
    """
    def __init__(self, ch):
        self.playerJustMoved = 2 # At the root pretend the player just moved is p2 - p1 has the first move
        self.chips = ch
        
    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = NimState(self.chips)
        st.playerJustMoved = self.playerJustMoved
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerJustMoved.
        """
        assert move >= 1 and move <= 3 and move == int(move)
        self.chips -= move
        self.playerJustMoved = 3 - self.playerJustMoved
        
    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return range(1,min([4, self.chips + 1]))
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        assert self.chips == 0
        if self.playerJustMoved == playerjm:
            return 1.0 # playerjm took the last chip and has won
        else:
            return 0.0 # playerjm's opponent took the last chip and has won

    def __repr__(self):
        s = "Chips:" + str(self.chips) + " JustPlayed:" + str(self.playerJustMoved)
        return s

class OXOState:
    """ A state of the game, i.e. the game board.
        Squares in the board are in this arrangement
        012
        345
        678
        where 0 = empty, 1 = player 1 (X), 2 = player 2 (O)
    """
    def __init__(self):
        self.playerJustMoved = 2 # At the root pretend the player just moved is p2 - p1 has the first move
        self.board = [0,0,0,0,0,0,0,0,0] # 0 = empty, 1 = player 1, 2 = player 2
        
    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OXOState()
        st.playerJustMoved = self.playerJustMoved
        st.board = self.board[:]
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        assert move >= 0 and move <= 8 and move == int(move) and self.board[move] == 0
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[move] = self.playerJustMoved
        
    def UndoMove(self, lastmove):
        """ Update a state by undoing out the given lastmove.
            Must update playerToMove.
        """
        assert lastmove >= 0 and lastmove <= 8 and lastmove == int(lastmove) and self.board[lastmove] != 0
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[lastmove] = 0
        
    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        for (x,y,z) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
            if self.board[x] == self.board[y] == self.board[z] != 0:
                return []
        return [i for i in range(9) if self.board[i] == 0]
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        for (x,y,z) in [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]:
            if self.board[x] == self.board[y] == self.board[z]:
                if self.board[x] == playerjm:
                    return 1.0
                else:
                    return 0.0
        if self.GetMoves() == []: return 0.5 # draw
        assert False # Should not be possible to get here

    def __repr__(self):
        s= ""
        for i in range(9): 
            s += ".XO"[self.board[i]] + " "
            if i % 3 == 2: s += "\n"
        return s

class OthelloState:
    """ A state of the game of Othello, i.e. the game board.
        The board is a 2D array where 0 = empty (.), 1 = player 1 (X), 2 = player 2 (O).
        In Othello players alternately place pieces on a square board - each piece played
        has to sandwich opponent pieces between the piece played and pieces already on the 
        board. Sandwiched pieces are flipped.
        This implementation modifies the rules to allow variable sized square boards and
        terminates the game as soon as the player about to move cannot make a move (whereas
        the standard game allows for a pass move). 
    """
    def __init__(self,sz = 8):
        self.playerJustMoved = 2 # At the root pretend the player just moved is p2 - p1 has the first move
        self.board = [] # 0 = empty, 1 = player 1, 2 = player 2
        self.size = sz
        assert sz == int(sz) and sz % 2 == 0 # size must be integral and even
        for y in range(sz):
            self.board.append([0]*sz)
        self.board[sz/2][sz/2] = self.board[sz/2-1][sz/2-1] = 1
        self.board[sz/2][sz/2-1] = self.board[sz/2-1][sz/2] = 2

    def Clone(self):
        """ Create a deep clone of this game state.
        """
        st = OthelloState()
        st.playerJustMoved = self.playerJustMoved
        st.board = [self.board[i][:] for i in range(self.size)]
        st.size = self.size
        return st

    def DoMove(self, move):
        """ Update a state by carrying out the given move.
            Must update playerToMove.
        """
        (x,y)=(move[0],move[1])
        assert x == int(x) and y == int(y) and self.IsOnBoard(x,y) and self.board[x][y] == 0
        m = self.GetAllSandwichedCounters(x,y)
        self.playerJustMoved = 3 - self.playerJustMoved
        self.board[x][y] = self.playerJustMoved
        for (a,b) in m:
            self.board[a][b] = self.playerJustMoved
    
    def GetMoves(self):
        """ Get all possible moves from this state.
        """
        return [(x,y) for x in range(self.size) for y in range(self.size) if self.board[x][y] == 0 and self.ExistsSandwichedCounter(x,y)]

    def AdjacentToEnemy(self,x,y):
        """ Speeds up GetMoves by only considering squares which are adjacent to an enemy-occupied square.
        """
        for (dx,dy) in [(0,+1),(+1,+1),(+1,0),(+1,-1),(0,-1),(-1,-1),(-1,0),(-1,+1)]:
            if self.IsOnBoard(x+dx,y+dy) and self.board[x+dx][y+dy] == self.playerJustMoved:
                return True
        return False
    
    def AdjacentEnemyDirections(self,x,y):
        """ Speeds up GetMoves by only considering squares which are adjacent to an enemy-occupied square.
        """
        es = []
        for (dx,dy) in [(0,+1),(+1,+1),(+1,0),(+1,-1),(0,-1),(-1,-1),(-1,0),(-1,+1)]:
            if self.IsOnBoard(x+dx,y+dy) and self.board[x+dx][y+dy] == self.playerJustMoved:
                es.append((dx,dy))
        return es
    
    def ExistsSandwichedCounter(self,x,y):
        """ Does there exist at least one counter which would be flipped if my counter was placed at (x,y)?
        """
        for (dx,dy) in self.AdjacentEnemyDirections(x,y):
            if len(self.SandwichedCounters(x,y,dx,dy)) > 0:
                return True
        return False
    
    def GetAllSandwichedCounters(self, x, y):
        """ Is (x,y) a possible move (i.e. opponent counters are sandwiched between (x,y) and my counter in some direction)?
        """
        sandwiched = []
        for (dx,dy) in self.AdjacentEnemyDirections(x,y):
            sandwiched.extend(self.SandwichedCounters(x,y,dx,dy))
        return sandwiched

    def SandwichedCounters(self, x, y, dx, dy):
        """ Return the coordinates of all opponent counters sandwiched between (x,y) and my counter.
        """
        x += dx
        y += dy
        sandwiched = []
        while self.IsOnBoard(x,y) and self.board[x][y] == self.playerJustMoved:
            sandwiched.append((x,y))
            x += dx
            y += dy
        if self.IsOnBoard(x,y) and self.board[x][y] == 3 - self.playerJustMoved:
            return sandwiched
        else:
            return [] # nothing sandwiched

    def IsOnBoard(self, x, y):
        return x >= 0 and x < self.size and y >= 0 and y < self.size
    
    def GetResult(self, playerjm):
        """ Get the game result from the viewpoint of playerjm. 
        """
        jmcount = len([(x,y) for x in range(self.size) for y in range(self.size) if self.board[x][y] == playerjm])
        notjmcount = len([(x,y) for x in range(self.size) for y in range(self.size) if self.board[x][y] == 3 - playerjm])
        if jmcount > notjmcount: return 1.0
        elif notjmcount > jmcount: return 0.0
        else: return 0.5 # draw

    def __repr__(self):
        s= ""
        for y in range(self.size-1,-1,-1):
            for x in range(self.size):
                s += ".XO"[self.board[x][y]]
            s += "\n"
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


def UCT(rootstate, itermax, verbose = False):
    """ Conduct a UCT search for itermax iterations starting from rootstate.
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
            state.DoMove(random.choice(state.GetMoves()))

        # Backpropagate
        while node != None: # backpropagate from the expanded node and work back to the root node
            node.Update(state.GetResult(node.playerJustMoved)) # state is terminal. Update node with result from POV of node.playerJustMoved
            node = node.parentNode

    # Output some information about the tree - can be omitted
    if (verbose): print(rootnode.TreeToString(0))
    else: print(rootnode.ChildrenToString())
                
def UCTPlayGame():
    """ Play a sample game between two UCT players where each player gets a different number 
        of UCT iterations (= simulations = tree nodes).
    """
    # state = OthelloState(4) # uncomment to play Othello on a square board of the given size
    state = OXOState() # uncomment to play OXO
    # state = NimState(15) # uncomment to play Nim with the given number of starting chips
    while (state.GetMoves() != []):
        print(repr(state))
        if state.playerJustMoved == 2:
            m = UCT(rootstate = state, itermax = 10, verbose = False) # play with values for itermax and verbose = True
        else:
            m = UCT(rootstate = state, itermax = 100, verbose = False)
        print("Best Move: " + str(m) + "\n")
        state.DoMove(m)
    if state.GetResult(state.playerJustMoved) == 1.0:
        print("Player " + str(state.playerJustMoved) + " wins!")
    elif state.GetResult(state.playerJustMoved) == 0.0:
        print("Player " + str(3 - state.playerJustMoved) + " wins!")
    else: print("Nobody wins!")

def UTCPlayVsHuman():
    """ Play a sample game between UCT player and human. Computer plays first.
        Number of UCT iterations is adjustable
    """
    playagain = "y"
    while playagain == "y":
        # state = OthelloState(4) # uncomment to play Othello on a square board of the given size
        state = OXOState() # uncomment to play OXO
        # state = NimState(15) # uncomment to play Nim with the given number of starting chips
        iterations = eval(input("How many UTC iterations does the Computer opponent get?\n(The more iterations the better it plays, but any \nnumber nabove 10000 will cause it to run slowly)\n"))
        while isinstance(iterations, int) != True or iterations < 0:
            iterations = eval(input("Error: Please input a positive integer.\n"))
        humanplayer = eval(input("Would you like to be player 1 or 2?\n"))
        while humanplayer != 1 and humanplayer != 2:
            humanplayer = eval(input("Error: Please input either 1 or 2.\n"))
        while (state.GetMoves() != []):
            print(repr(state))
            if state.playerJustMoved == humanplayer:
                m = UCT(rootstate = state, itermax = iterations, verbose = False)
            else:
                m = eval(input("Where would you like to place your piece? \n0 1 2\n3 4 5\n6 7 8\n")) # Change if not playing OXO
                while state.board[m] != 0:
                    m = eval(input("Error: Please pick an empty location.\nThese are the remaining empty locations:\n" + str(state.GetMoves()) + "\n"))
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
import numpy as np

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
    state = OXOState() # uncomment to play OXO
    # state = NimState(15) # uncomment to play Nim with the given number of starting chips
    boards = []
    while (state.GetMoves() != []):
        m = MUCT(rootstate = state, itermax = 12, verbose = False) # play with values for itermax and verbose = True
        state.DoMove(m)
        boards.append(state.board)
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
    print(training_data)
    EvalNetwork.update_mini_batch(training_data, eta)

def Train(practice_sessions, minibatch_size, eta):
	for s in range(practice_sessions):
		Practice_Session(minibatch_size, eta)
		print("practice session {0} completed\n".format(s))
	print("training complete\n")

def MUCTPlayHuman():
        """ Play a sample game between MUCT player and human. Computer plays first.
        Number of MUCT iterations is adjustable
        """
        playagain = "y"
        while playagain == "y":
                # state = OthelloState(4) # uncomment to play Othello on a square board of the given size
                state = OXOState() # uncomment to play OXO
                # state = NimState(15) # uncomment to play Nim with the given number of starting chips
                iterations = eval(input("How many UTC iterations does the Computer opponent get?\n(The more iterations the better it plays, but any \nnumber nabove 10000 will cause it to run slowly)\n"))
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
                        m = eval(input("Where would you like to place your piece? \n0 1 2\n3 4 5\n6 7 8\n")) # Change if not playing OXO
                while state.board[m] != 0:
                        m = eval(input("Error: Please pick an empty location.\nThese are the remaining empty locations:\n" + str(state.GetMoves()) + "\n"))
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
        networksizes = [9,27,1]# [9, hiddenlayerneurons, 1]
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
