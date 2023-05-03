import tensorflow as tf

def expected_mean_payoffs_tensor(W,S0,S1,iters=1000,tol=1e-6):
    R=[]
    for d0,d1 in ([S1,S0],[S0,S1]):
        d=d0@d1
        a=tf.reduce_sum(W*d0,axis=-1,keepdims=True)
        b=tf.reduce_sum(W*d1,axis=-1,keepdims=True)
        q=d0@b+a
        D=tf.eye(W.shape[-1],batch_shape=W.shape[:-2])
        S=tf.zeros(W.shape)
        if iters is not None:
            for i in range(iters):
                S+=D
                D=D@d
            R.append((S@q)/(2*iters))
        else:
            # Perform the eigen decomposition of the one turn transition matrix
            D,V=tf.linalg.eig(d)
            # Compute the mean payoff projection matrix
            # It is a projection matrix into the 1-eigenspace of the one turn transition matrix parallel to the complement eigenspaces
            D_mean=tf.where(tf.abs(D-1)<=tol,tf.constant([1.+0.j],dtype=tf.complex64),tf.constant([0.+0.j],dtype=tf.complex64))
            # Compute the mean payoffs
            Q=tf.math.real(V@tf.linalg.diag(D_mean)@tf.linalg.inv(V))
            R.append(Q@q/2)
    return tf.concat(R,axis=-1)


def mat_power(A,n):
    """
    Compute the matrix power A^n
    :param A: The matrix
    :param n: The power
    """
    if n==0:
        return tf.eye(A.shape[-1],batch_shape=A.shape[:-2])
    elif n==1:
        return A
    elif n%2==0:
        return mat_power(A@A,n//2)
    else:
        return A@mat_power(A@A,n//2)

def expected_discounted_payoffs_tensor(W,S0,S1,gamma,iters=1000,normalize=False):
    R=[]
    if tf.abs(gamma)>=1:
        raise ValueError("The discount factor must be in (-1,1)")
    for d0,d1 in ([S1,S0],[S0,S1]):
        d=d0@d1
        a=tf.reduce_sum(W*d0,axis=-1,keepdims=True)
        b=tf.reduce_sum(W*d1,axis=-1,keepdims=True)
        q=d0@b+a
        Id=tf.eye(W.shape[-1],batch_shape=W.shape[:-2])
        A=tf.linalg.inv(Id-gamma*d)
        if iters is not None:
            B=Id-mat_power(gamma*d,iters)
            z=A@(B@q)/2
            if normalize:
                z*=(1-gamma)
            R.append(z)
        else:
            z=A@q/2
            if normalize:
                z*=(1-gamma)
            R.append(z)
    return tf.concat(R,axis=-1)
