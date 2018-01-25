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

def Practice_Session(minibatch_size, eta):
	training_data = []
	for i in range(minibatch_size)
		d = MUTCPlayGame()
		training_data.append(d)
	update_mini_batch(training_data, eta)

def Train(practice_sessions, minibatch_size, eta, network_sizes):
	net.Network(network_sizes)
	for s in practice_sessions:
		Practice_Session(minibatch_size, eta)
		print("practice session {0} completed\n".format(s))
	print("training complete\n")
	
if __name__ == "__main__":
    """ Play as human v the MUCT player 
    """
	MUCTPlayHuman()

"""
IDEAS FOR FUTURE IMPROVEMENTS
~~~~~~~~~~~

- Would like to include method for storing the trained network in a file.
- Would like to create interface for playing the games--probably importing a different file.
"""
