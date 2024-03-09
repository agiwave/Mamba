from .. import numpy as provider
from .Bases import Parameter
from .Containers import Parrallel, Functional

def _LROperator(oper, left, right=None):
    if(right == None):
        return Functional(oper, left)
    else:
        return Parrallel(left, right, join_module=oper)

# Functions
def Add(left, right=None): return _LROperator(provider.add, left, right)
def Sub(left, right=None): return _LROperator(provider.sub, left, right)
def Mul(left, right=None): return _LROperator(provider.mul, left, right)
def Div(left, right=None): return _LROperator(provider.div, left, right)
def Dot(left, right=None): return _LROperator(provider.dot, left, right)
def MatMul(left, right=None): return _LROperator(provider.matmul, left, right)

def Scale(data): return Functional(provider.mul,Parameter(data))
def Resident(block): return Parrallel(provider.iden, block, join_module=provider.add)