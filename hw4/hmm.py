import numpy as np


def forward(A, B, pi, O):
    """
    Calculates the probability of an observation sequence O given the model(A, B, pi).
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities (N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The probability of the observation sequence, and the calculated alphas in the Trellis diagram with shape
             (N, T) which should be a numpy array.
    :return prob, Trellis diagram
    """

    N, M = B.shape
    T = O.shape[0]

    trellis_diagram = np.zeros((N,T))

    #iterates thru emmisions ----> O(T)
    for j in range(0, T):
        #iterates thru states -----> O(N)
        for i in range(0, N): 
            #first emit: init case, trellis_diagram[i][0] = Pi[i]*B[i,O[0]]
            if j == 0:
                trellis_diagram[i][0] = pi[i]*B[i][O[0]]
            else:
                accum = 0
                #check for all possible transitions from previous states ------------> O(N)
                for k in range(0, N):
                    accum += (trellis_diagram[k][j-1]*A[k][i])

                #add emmision prob for all possible transitions, which are all the same, so simply multiply 
                accum *= B[i][O[j]]
                trellis_diagram[i][j] = accum

    res_prob =0
    for state_index in range(0,N):
        res_prob += trellis_diagram[state_index][T-1]



    return res_prob, trellis_diagram








def viterbi(A, B, pi, O):
    """
    Calculates the most likely state sequence given model(A, B, pi) and observation sequence.
    :param A: state transition probabilities (NxN)
    :param B: observation probabilites (NxM)
    :param pi: initial state probabilities(N)
    :param O: sequence of observations(T) where observations are just indices for the columns of B (0-indexed)
        N is the number of states,
        M is the number of possible observations, and
        T is the sequence length.
    :return: The most likely state sequence with shape (T,) and the calculated deltas in the Trellis diagram with shape
             (N, T). They should be numpy arrays.
    """

    N, M = B.shape
    T = O.shape[0]

    seq = []

    trellis_diagram = np.zeros((N,T))

    #iterates thru emmisions ----> O(T)
    for j in range(0, T):
        most_likely_state = None
        most_likely_state_alpha = -1        
        #iterates thru states -----> O(N)
        for i in range(0, N): 

            #first emit: init case, trellis_diagram[i][0] = Pi[i]*B[i,O[0]]
            if j == 0:
                trellis_diagram[i][0] = pi[i]*B[i][O[0]]
                if trellis_diagram[i][0] > most_likely_state_alpha:
                    most_likely_state_alpha = trellis_diagram[i][0]
                    most_likely_state = i

            else:
                max_edge_val = -1
                #corresponding_alpha = None
                #check for all possible transitions from previous states ------------> O(N)
                for k in range(0, N):
                    cond = trellis_diagram[k][j-1]*A[k][i]*B[i][O[j]]
                    if cond > max_edge_val:
                        max_edge_val = cond
                        #corresponding_alpha = k
                trellis_diagram[i][j] = max_edge_val
            
                if max_edge_val > most_likely_state_alpha:
                    most_likely_state_alpha = max_edge_val
                    most_likely_state = i
        seq.append(most_likely_state)
    return seq, trellis_diagram