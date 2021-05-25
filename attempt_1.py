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


f = 1 # factor to apply to mu

'''
suffix = '_2A' # Fig 2A
h = 0       # homophily
r = 0.2     # risk
beta = 3

suffix = '_2B' # for this one, the attractor is actually in the middle if you increase mu (1.2*)
h = 0.7     # homophily
r = 0.2     # risk
beta = 3

suffix = '_2C' # this one's grad looks different to what they got, and when I increase mu the stationary distn moves to the middle (1.8*)
h = 1       # homophily
r = 0.2     # risk -- if I decrease this to 0.1 I get something more like what they got
beta = 3

suffix = '_2D'
h = 0       # homophily
r = 0.3     # risk
beta = 3

suffix = '_2E'
h = 0.7     # homophily
r = 0.3     # risk
beta = 3

suffix = '_2F' # this one is weird but increase mu and stops having stationary distn in bottom left corner
h = 1.0     # homophily
r = 0.3     # risk
beta = 3

suffix = '_2C_explore' # this one's grad looks different to what they got, and when I increase mu the stationary distn moves to the middle (1.8*)
h = 1       # homophily
r = 0.1     # risk -- if I decrease this to 0.1 I get something more like what they got
beta = 3
f = 1.8

suffix = '_2B_explore' # I wonder if the sudden loss of cooperation is really just an edge effect
h = 0.7
r = 0.2
beta = 3
f = 1.4 # yes. I increase the mutation rate a bit here, and the cooperation is recovered close to what it was in 2A
'''

suffix = '_2A_explore' # increase its mutation rate as well so I can do a more direct comparison to 2B
h = 0       # homophily
r = 0.2     # risk
beta = 3
f = 1.4

'''
suffix = '_spare' # this one is weird but increase mu and stops having stationary distn in bottom left corner
h = 1.0     # homophily
r = 0.3     # risk
beta = 3
f = 1
'''

# TODO I should find something that looks like their 2B with two attractors

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


# define handy functions
# ---

# indicator function for whether or not the group has met or exceeded the threshold
# a function of the number of rich cooperators jR and poor cooperators jP in the group
Theta_fnc = lambda jR, jP: 1 if cR*jR + cP*jP - M*c*cbar >= 0 else 0
    
# payoffs
PiDR = lambda jR, jP: bR * ( Theta_fnc(jR, jP) + (1-r)*(1-Theta_fnc(jR, jP)) )      # rich defectors
PiDP = lambda jR, jP: bP * ( Theta_fnc(jR, jP) + (1-r)*(1-Theta_fnc(jR, jP)) )      # poor defectors
PiCR = lambda jR, jP: bR * ( Theta_fnc(jR, jP) + (1-r)*(1-Theta_fnc(jR, jP)) ) - cR # rich cooperators
PiCP = lambda jR, jP: bP * ( Theta_fnc(jR, jP) + (1-r)*(1-Theta_fnc(jR, jP)) ) - cP # poor cooperators

# fitness expected values are a function of the composition of the population (rich and poor cooperators)

# rich cooperators
fCR = lambda iR, iP: (1/comb(Z-1, N-1)) * \
        sum( sum(  
            comb(iR-1, jR) * comb(iP, jP) * comb(Z-iR-iP, N-1-jR-jP) * PiCR(jR+1, jP) 
            for jP in range(N-jR) ) for jR in range(N) )
es

# poor cooperators
fCP = lambda iR, iP: (1/comb(Z-1, N-1)) * \
        sum( sum(  
            comb(iR, jR) * comb(iP-1, jP) * comb(Z-iR-iP, N-1-jR-jP) * PiCP(jR, jP+1) 
            for jP in range(N-jR) ) for jR in range(N) )

# rich defectors
fDR = lambda iR, iP: (1/comb(Z-1, N-1)) * \
        sum( sum(  
            comb(iR, jR) * comb(iP, jP) * comb(Z-1-iR-iP, N-1-jR-jP) * PiDR(jR, jP) 
            for jP in range(N-jR) ) for jR in range(N) )

# poor defectors
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
# chance that the candidate will imitate the other's strategy is proportional to how much higher the other's payoff is compared to their own
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


# enumerate all possible states (iR, iP) i.e., (no rich cooperators, no poor cooperators)
# ---

iV = [ (iR, iP) for iP in range(ZP+1) for iR in range(ZR+1) ]   # list of states
i2idx = { i: idx for idx, i in enumerate(iV) }                  # reverse dictionary from state to index of W below
len_iV = len(iV)


# create W, grad
# ---

# transition probability matrix between population states (iR, iP) -> (iR', iP')
W = np.zeros( (len_iV, len_iV) )

# gradient of selection on rich and poor axis
grad_iR = np.zeros( (ZP+1, ZR+1) ) # rich on the x-axis
grad_iP = np.zeros( (ZP+1, ZR+1) )

# probability to not transition from this population state (composition stays where it is)
stay_iR = np.zeros( (ZP+1, ZR+1) )
stay_iP = np.zeros( (ZP+1, ZR+1) )

# probability to transition from this state (in any direction)
go_iR = np.zeros( (ZP+1, ZR+1) )
go_iP = np.zeros( (ZP+1, ZR+1) )

for idx, i in enumerate(iV):

    iR, iP = i  # number of rich cooperators, number of poor cooperators

    # each state can only ever transition in one of four ways:
    #   iR -> iR+1, iR -> iR-1, iP -> iP+1, iP -> iP-1, 
    # i.e., rich cooperators increase by 1, rich cooperators decrease by 1, poor cooperators increase by 1, poor cooperators decrease by 1,
    # or it stays the same

    # calculate probabilities of each transition and population the matrix W

    if iR < ZR:
        TiR_gain = T(iR, iP, 'D', 'R')
        W[i2idx[(iR+1, iP)], idx] = TiR_gain
    else:
        TiR_gain = 0 # not possible to have more rich cooperators than rich individuals in the population

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

    # probability that the population state does not transition from (iR, iP)
    W[(idx, idx)] = 1 - TiR_gain - TiR_loss - TiP_gain - TiP_loss

    # gradients used for a quiver plot below
    grad_iR[iP, iR] = TiR_gain - TiR_loss
    grad_iP[iP, iR] = TiP_gain - TiP_loss

    # store these as well out of curiosity
    stay_iR[iP, iR] = 1 - TiR_gain - TiR_loss
    stay_iP[iP, iR] = 1 - TiP_gain - TiP_loss

    go_iR[iP, iR] = TiR_gain + TiR_loss
    go_iP[iP, iR] = TiP_gain + TiP_loss


# get relative proportions of time spent in different states
# ---

eigs, leftv, rightv = eig(W, left=True, right=True)
domIdx = np.argmax( np.real(eigs) ) # index of the dominant eigenvalue
L = np.real( eigs[domIdx] )         # the dominant eigenvalue
p = np.real(rightv[:, domIdx])      # the right-eigenvector is the relative proportions in classes at steady state
p = p / np.sum(p)                   # normalise it

# populate a big matrix with the p values
# ---

# out of curiosity, option to plot it below

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
plt.savefig('attempt' + suffix + '.pdf')
plt.close()
