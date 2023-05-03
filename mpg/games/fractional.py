import numpy as np

def expected_mean_payoffs(W,S0,S1,iters=1000,tol=1e-6):
    """
    Compute the expected mean payoffs of a game with a pair of fractional strategies.
    A fractional strategy is a matrix with entries in [0,1] that sum to 1 along each row.
    :param W: The payoff matrix
    :param S0: The first fractional strategy
    :param S1: The second fractional strategy
    :param iters: The number of iterations to use in the power method. If None, use the eigenvector method.
    :param tol: The tolerance for the power method
    """
    R=[]
    for d0,d1 in ([S1,S0],[S0,S1]):
        n=len(W)
        d=d0@d1
        a=np.sum(W*d0,axis=-1)
        b=np.sum(W*d1,axis=-1)
        q=d0@b+a
        D=np.eye(n)
        S=np.zeros([n,n])
        if iters is not None:
            for i in range(iters):
                S+=D
                D=D@d
            R.append((S@q)/(2*iters))
        else:
            D,V=np.linalg.eig(d)
            D_mean=np.where(np.abs(D-1)<=tol,1,0)
            Q=V@np.diag(D_mean)@np.linalg.inv(V)
            R.append(Q@q/2)
    return R

def expected_discounted_payoffs(W,S0,S1,gamma,iters=1000,tol=1e-6):
    """
    Compute the expected discounted payoffs of a game with a pair of fractional strategies.
    A fractional strategy is a matrix with entries in [0,1] that sum to 1 along each row.
    :param W: The payoff matrix
    :param S0: The first fractional strategy
    :param S1: The second fractional strategy
    :param iters: The number of iterations to use in the power method. If None, use the eigenvector method.
    :param tol: The tolerance for the power method
    """
    if np.abs(gamma)>=1:
        raise ValueError("The discount factor must be in (-1,1)")
    R=[]
    for d0,d1 in ([S1,S0],[S0,S1]):
        n=len(W)
        d=d0@d1
        a=np.sum(W*d0,axis=-1)
        b=np.sum(W*d1,axis=-1)
        q=d0@b+a
        R.append(np.linalg.inv(np.eye(n)-gamma*d)@q/2)
    return R