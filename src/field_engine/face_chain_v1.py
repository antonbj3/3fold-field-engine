"""Exact axis-face certificates for products of RSOC3: a>=0,c>=0,b²<=ac.
No discovery, no general-cone completeness. Unchecked/malformed input => UNKNOWN.
Each step proves a block's a or c zero using current equalities, then adds b=0.
Terminal equality combination must be 0=nonzero. Fractions/ints/strings only.
"""
from fractions import Fraction

def _q(x):
    if isinstance(x,bool) or not isinstance(x,(int,str,Fraction)):
        raise ValueError('exact rational required')
    return Fraction(x)

def _comb(A,b,y):
    if len(y)!=len(A): raise ValueError('multiplier length')
    y=list(map(_q,y));n=len(A[0])
    return [sum((y[i]*A[i][j] for i in range(len(A))),Fraction()) for j in range(n)],sum((u*v for u,v in zip(y,b)),Fraction())

def verify(A,b,blocks,steps,terminal):
    try:
        if not A or not A[0]: raise ValueError('empty matrix')
        A=[list(map(_q,row)) for row in A];b=list(map(_q,b));n=len(A[0])
        if len(A)!=len(b) or any(len(row)!=n for row in A): raise ValueError('dimensions')
        flat=[i for block in blocks for i in block]
        if any(len(block)!=3 for block in blocks) or any(type(i)!=int for i in flat) or sorted(flat)!=list(range(n)):
            raise ValueError('blocks must partition all columns')
        trace=[]
        for step in steps:
            idx=step['block'];side=step['side']
            if type(idx)!=int or not 0<=idx<len(blocks) or side not in ('a','c'): raise ValueError('face selector')
            a,mid,c=blocks[idx];coord=a if side=='a' else c
            row,rhs=_comb(A,b,step['y'])
            if rhs!=0 or row!=[Fraction(int(j==coord)) for j in range(n)]: raise ValueError('unproved zero face')
            A.append([Fraction(int(j==mid)) for j in range(n)]);b.append(Fraction())
            trace.append({'block':idx,'zero_coordinate':coord,'implied_zero':mid})
        row,rhs=_comb(A,b,terminal)
        if any(row) or rhs==0: raise ValueError('no terminal contradiction')
        return {'verdict':'NO','trace':trace,'terminal_rhs':str(rhs),'scope':'closed RSOC product, supplied equality system'}
    except (ValueError,TypeError,KeyError,ZeroDivisionError,IndexError,OverflowError):
        return {'verdict':'UNKNOWN'}

def fixture(n):
    if type(n)!=int or n<1: raise ValueError('n')
    blocks=[(3*i,3*i+1,3*i+2) for i in range(n)]
    A=[[int(j==0) for j in range(3*n)]];b=[0]
    for i in range(n-1):
        A.append([int(j==3*(i+1))-int(j==3*i+1) for j in range(3*n)]);b.append(0)
    A.append([int(j==3*(n-1)+1) for j in range(3*n)]);b.append(1)
    steps=[]
    for i in range(n):
        y=[0]*(n+1+i)
        if i==0:y[0]=1
        else:y[i]=1;y[n+i]=1
        steps.append({'block':i,'side':'a','y':y})
    y=[0]*(2*n+1);y[n]=1;y[-1]=-1
    return dict(A=A,b=b,blocks=blocks,steps=steps,terminal=y)
