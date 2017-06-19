import numpy as np
import sys, traceback


class MoveEngine(object):
    """MoveEngine provides perturbation functions for the context during the NCMC
    simulation.
    Ex.
        from blues.ncmc import MoveEngine
        probabilities = [0.25, 0.75]
        #Where move is a list of two Move objects
        mover = MoveEngine(move, probabilities)
        #Get the dictionary of proposed moves
        mover.moves
    """
    def __init__(self, moves, probabilities=None):
        """Initialize the MovePropsal object that contains functions to perturb
        the context in the NCMC simulation.
        Parameters
        ----------
        moves : blues.ncmc.Model object or list of n blues.ncmc.Model-like objects
            Specifies the possible moves to be performed.

        probabilities: list of floats, optional, default=None
            A list of n probabilities,
            where probabilities[i] corresponds to the probaility of moves[i]
            being selected to perform its associated move() method.

            If None, uniform probabilities are assigned.
        """

    #make a list from moves if not a list
        if isinstance(moves,list):
            self.moves = moves
        else:
            self.moves = [moves]
        #normalize probabilities
        if probabilities is None:
            single_prob = 1. / len(self.moves)
            self.probs = [single_prob for x in (self.moves)]
        else:
            prob_sum = float(sum(probabilities))
            self.probs = [x/prob_sum for x in probabilities]
        #if move and probabilitiy lists are different lengths throw error
        if len(self.moves) != len(self.probs):
            print('moves and probability list lengths need to match')
            raise IndexError
        #use index in selecting move
        self.selected_move = None

    def selectMove(self):
        """chooses the move which will be selected for a given NCMC
        iteration
        """
        rand_num = np.random.choice(len(self.probs), p=self.probs)
        self.selected_move = rand_num

    def runEngine(self, context):
        """Selects a random Move object based on its
        assigned probability and and performs its move() function
        on a context.
        Parameters
        ----------
        context : openmm.context object
        OpenMM context whose positions should be moved.
        """
        try:
            new_context = self.moves[self.selected_move].move(context)
        except Exception as e:
            #In case the move isn't properly implemented, print out useful info
            print('Error: move not implemented correctly, printing traceback:')
            ex_type, ex, tb = sys.exc_info()
            traceback.print_tb(tb)
            print(e)
            raise SystemExit

        return new_context