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

import network.py as net
import MCTS.py 

def softmax(values):
	return np.exp(values)/sum(exp(values))

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
        	newboards = []
        	for move in state.GetMoves():
        		state.DoMove(move)
        		newboards.append(state.board)
        		state.UndoMove(move)
        	weights = softmax(EvalNetwork.feedforward(newboards[i]) for i in range(len(state.GetMoves())))
        	rollmov = np.random.choices(state.GetMoves(), weights = weights)
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
    while (state.GetMoves() != []):
        m = MUCT(rootstate = state, itermax = 100, verbose = False) # play with values for itermax and verbose = True
        state.DoMove(m)
    if state.GetResult(state.playerJustMoved) == 1.0:
        print("Player " + str(state.playerJustMoved) + " wins!")
    elif state.GetResult(state.playerJustMoved) == 0.0:
        print("Player " + str(3 - state.playerJustMoved) + " wins!")
    else: print("Nobody wins!")

def Practice_Session(minibatch_size, eta):
	training_data = []
	for i in range(minibatch_size)
		d = MUTCPlayGame()
		training_data.append(d)
	update_mini_batch(training_data, eta)

def Train(practice_sessions, minibatch_size, eta, network_sizes):
	EvalNetwork = net.Network(network_sizes)
	for s in practice_sessions:
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
	hiddenlayerneurons = eval(input("How many hidden layer neurons for this model?\n"))
	while isinstance(hiddenlayerneurons, int) != True or hiddenlayerneurons < 0:
		hiddenlayerneurons = eval(input("Error: Please input a positive integer.\n"))  
	networksizes = [9, hiddenlayerneurons, 1]
	eta = eval(input("What learning rate would you like to use?\n"))
	while isinstance(eta, float) != True or eta < 0 or eta > 0.5:
		eta = eval(input("Error: Please input a positive number less than 0.5.\n"))
	minibatchsize = eval(input("How many games per practice session?\n"))
	while isinstance(minibatchsize, int) != True or minibatchsize < 0:
		minibatchsize = eval(input("Error: Please input a positive integer.\n"))
	numsessions = eval(input("How many practice sessions to complete training?\n"))
	while isinstance(numsessions, int) != True or numsessions < 0:
		numsessions = eval(input("Error: Please input a positive integer.\n"))
	Train(numsessions, minibatchsize, eta, networksizes)
	MUCTPlayHuman()

"""
IDEAS FOR FUTURE IMPROVEMENTS
~~~~~~~~~~~
- Would like to include method for storing the trained network in a file.
- Would like to create interface for playing the games--probably importing a different file.
"""
