
def iden(x, *args):
    if(len(args)==0):
        return x
    else:
        return x, *args
        
def Iden(*args): 
    return iden
    