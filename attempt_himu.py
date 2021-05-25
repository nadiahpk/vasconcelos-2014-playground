# Try to replicate some of Vasconcelos et al 2014 
# NOTE to self: the reason the p values sometimes don't line up with the gradient field is because the corners and edges are sticky
# homophily makes the edges sticky

from scipy.special import comb as comb
#import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig as eig


# parameters
# ---


f = 3 # factor to apply to mu
suffixV = [ '_2A', '_2B', '_2C', '_2D', '_2E', '_2F' ]
hV =      [     0,   0.7,     1,     0,   0.7,    1  ]
rV =      [   0.2,   0.2,   0.2,   0.3,   0.3,   0.3 ]

# TODO I should find something that looks like their 2B with two attractors

beta = 3

Z = 200     # population, rich, poor
ZR = 40
ZP = 160 
N = 6       # game group size

# DEBUGGING NOTE reduce the size for the moment
Z = 20; ZR = 4; ZP = 16
Z = 40; ZR = 8; ZP = 32

c = 0.1     # contribution to PGG
bP = 0.625  # endowments
bR = 2.5
M = 3       # I'm assuming that's a typo

# calculate secondary parameters
# ---

cR = c*bR
cP = c*bP

cbar = (ZR*bR + ZP*bP) / Z # == 1

mu = f*1/Z

for suffix, h, r in zip(suffixV, hV, rV):



    # define handy functions
    # ---

    # Theta_fnc = lambda Delta: 1 if Delta >= 0 else 0
    # Delta_fnc = lambda jR, jP: cR*jR + cP*jP - M*c*cbar
    Theta_fnc = lambda jR, jP: 1 if cR*jR + cP*jP - M*c*cbar >= 0 else 0
        
    # payoffs
    PiDR = lambda jR, jP: bR * ( Theta_fnc(jR, jP) + (1-r)*(1-Theta_fnc(jR, jP)) )
    PiDP = lambda jR, jP: bP * ( Theta_fnc(jR, jP) + (1-r)*(1-Theta_fnc(jR, jP)) )
    PiCR = lambda jR, jP: bR * ( Theta_fnc(jR, jP) + (1-r)*(1-Theta_fnc(jR, jP)) ) - cR
    PiCP = lambda jR, jP: bP * ( Theta_fnc(jR, jP) + (1-r)*(1-Theta_fnc(jR, jP)) ) - cP

    # fitness
    fCR = lambda iR, iP: (1/comb(Z-1, N-1)) * \
            sum( sum(  
                comb(iR-1, jR) * comb(iP, jP) * comb(Z-iR-iP, N-1-jR-jP) * PiCR(jR+1, jP) 
                for jP in range(N-jR) ) for jR in range(N) )

    fCP = lambda iR, iP: (1/comb(Z-1, N-1)) * \
            sum( sum(  
                comb(iR, jR) * comb(iP-1, jP) * comb(Z-iR-iP, N-1-jR-jP) * PiCP(jR, jP+1) 
                for jP in range(N-jR) ) for jR in range(N) )

    fDR = lambda iR, iP: (1/comb(Z-1, N-1)) * \
            sum( sum(  
                comb(iR, jR) * comb(iP, jP) * comb(Z-1-iR-iP, N-1-jR-jP) * PiDR(jR, jP) 
                for jP in range(N-jR) ) for jR in range(N) )

    fDP = lambda iR, iP: (1/comb(Z-1, N-1)) * \
            sum( sum(  
                comb(iR, jR) * comb(iP, jP) * comb(Z-1-iR-iP, N-1-jR-jP) * PiDP(jR, jP) 
                for jP in range(N-jR) ) for jR in range(N) )

    def fXk(iR, iP, X, k):

        if X == 'C' and k == 'R':
            res = fCR(iR, iP)
        elif X == 'C' and k == 'P':
            res = fCP(iR, iP)
        elif X == 'D' and k == 'R':
            res = fDR(iR, iP)
        elif X == 'D' and k == 'P':
            res = fDP(iR, iP)
        else:
            res = None

        return(res)

    # Fermi function
    Fe = lambda iR, iP, X1, k1, X2, k2: 1 + np.exp( beta*( fXk(iR, iP, X1, k1) - fXk(iR, iP, X2, k2) ) )

    def T(iR, iP, X, k):
        '''
        Probability of transition of a k-wealth individual (k in {R, P}) from strategy
        X (X in {C, D}) to Y (opposite of X)
        '''

        Y = 'C' if X == 'D' else 'D'
        l = 'R' if k == 'P' else 'P'

        Zk = ZR if k == 'R' else ZP
        Zl = ZR if l == 'R' else ZP

        ik = iR if k == 'R' else iP
        iXk = ik if X == 'C' else Zk - ik

        il = iR if l == 'R' else iP
        iYl = il if Y == 'C' else Zl - il
        iYk = ik if Y == 'C' else Zk - ik

        TXYk = (iXk/Z) * ( mu + (1-mu) * ( 
            iYk / ( (Zk - 1 + (1-h)*Zl) * Fe(iR, iP, X, k, Y, k) ) + \
            (1-h)*iYl / ( (Zk - 1 + (1-h)*Zl) * Fe(iR, iP, X, k, Y, l) )
            ))

        return(TXYk)


    # enumerate all possible states (iR, iP)
    # ---

    iV = [ (iR, iP) for iP in range(ZP+1) for iR in range(ZR+1) ]   # list of states
    i2idx = { i: idx for idx, i in enumerate(iV) }              # reverse dictionary from state to index
    len_iV = len(iV)


    # create W, grad
    # ---

    W = np.zeros( (len_iV, len_iV) )
    grad_iR = np.zeros( (ZP+1, ZR+1) ) # rich on the x-axis
    grad_iP = np.zeros( (ZP+1, ZR+1) )
    stay_iR = np.zeros( (ZP+1, ZR+1) ) # rich on the x-axis
    stay_iP = np.zeros( (ZP+1, ZR+1) )
    go_iR = np.zeros( (ZP+1, ZR+1) ) # rich on the x-axis
    go_iP = np.zeros( (ZP+1, ZR+1) )

    # each state can only ever transition in one of four ways:
    # iR -> iR+1, iR -> iR-1, iP -> iP+1, iP -> iP-1, 
    # or it stays the same

    for idx, i in enumerate(iV):

        iR, iP = i

        # calculate probabilities of each and population the matrix

        if iR < ZR:
            TiR_gain = T(iR, iP, 'D', 'R')
            W[i2idx[(iR+1, iP)], idx] = TiR_gain
        else:
            TiR_gain = 0

        if iR > 0:
            TiR_loss = T(iR, iP, 'C', 'R')
            W[i2idx[(iR-1, iP)], idx] = TiR_loss
        else:
            TiR_loss = 0

        if iP < ZP:
            TiP_gain = T(iR, iP, 'D', 'P')
            W[i2idx[(iR, iP+1)], idx] = TiP_gain
        else:
            TiP_gain = 0

        if iP > 0:
            TiP_loss = T(iR, iP, 'C', 'P')
            W[i2idx[(iR, iP-1)], idx] = TiP_loss
        else:
            TiP_loss = 0

        W[(idx, idx)] = 1 - TiR_gain - TiR_loss - TiP_gain - TiP_loss

        grad_iR[iP, iR] = TiR_gain - TiR_loss
        grad_iP[iP, iR] = TiP_gain - TiP_loss

        stay_iR[iP, iR] = 1 - TiR_gain - TiR_loss
        stay_iP[iP, iR] = 1 - TiP_gain - TiP_loss

        go_iR[iP, iR] = TiR_gain + TiR_loss
        go_iP[iP, iR] = TiP_gain + TiP_loss


    # get relative proportions of time spent in different states
    # ---

    eigs, leftv, rightv = eig(W, left=True, right=True)
    domIdx = np.argmax( np.real(eigs) ) # index of the dominant eigenvalue
    L = np.real( eigs[domIdx] )         # the dominant eigenvalue
    p = np.real(rightv[:, domIdx])      # the right-eigenvector is the relative proportions in classes at ss
    p = p / np.sum(p)                   # normalise it

    # populate a big matrix with the p values

    P = np.zeros( (ZP+1, ZR+1) ) # rich on the x-axis
    for idx, pi in enumerate(p):

        iR, iP = iV[idx]
        P[iP, iR] = pi


    # plot
    # ---

    fig, ax = plt.subplots(figsize=(3,6))

    iRV = list(range(ZR+1))
    iPV = list(range(ZP+1))

    #ax.pcolor(iRV, iPV, P, cmap='coolwarm_r', alpha=0.5) #, vmin=0, vmax=1)
    #im = ax.imshow(P, extent=(0-0.5, ZR+0.5, 0-0.5, ZP+0.5), origin='lower', cmap='coolwarm_r') #, alpha=0.5) #, vmin=0, vmax=1)
    # im = ax.imshow(P, extent=(0-0.5, ZR+0.5, 0-0.5, ZP+0.5), origin='lower', cmap='coolwarm_r')
    im = ax.imshow(P, origin='lower', cmap='coolwarm_r', alpha=0.5)
    #fig.colorbar(im)
    ax.quiver(iRV, iPV, grad_iR, grad_iP)

    #ax.quiver(iRV, iPV, np.zeros( (ZP+1, ZR+1) ), grad_iP)
    #ax.quiver(iRV, iPV, grad_iR, np.zeros( (ZP+1, ZR+1) ))

    #ax.quiver(iRV, iPV, stay_iR, np.zeros( (ZP+1, ZR+1) ))
    #ax.quiver(iRV, iPV, np.zeros( (ZP+1, ZR+1) ), stay_iP)

    #ax.quiver(iRV, iPV, go_iR, np.zeros( (ZP+1, ZR+1) ))
    #ax.quiver(iRV, iPV, np.zeros( (ZP+1, ZR+1) ), go_iP)


    ax.set_xlim( (-1, ZR+1) )
    ax.set_ylim( (-1, ZP+1) )
    #ax.set_aspect('scaled')
    ax.set_xlabel(r'rich cooperators, $i_R$')
    ax.set_ylabel(r'poor cooperators, $i_P$')
    plt.axis('scaled')
    plt.tight_layout()
    plt.savefig('attempt' + suffix + '_highmu.pdf')
    plt.close()
