# import sys
# ONLINE_JUDGE = __debug__
# if ONLINE_JUDGE:
#     import io,os
#     input = io.BytesIO(os.read(0,os.fstat(0).st_size)).readline        #fast input/output

# sys.stdout = open("in.txt")

# def inp():
#     return(int(input()))
# def inlt():
#     return(list(map(int,input().split())))
# def insr():
#     return(input().strip())
# def invr():
#     return(map(int,input().split()))

# sys.stdout.write(str(ans)+"\n")

def sieve(n,lst):                      # Sieve of Eratosthenes
    is_prime=[True]*(n+1)
    is_prime[1]=False
    for i in range(2,1000+10):
        if(is_prime[i]):
            for j in range(2*i, n+1, i):
                 is_prime[j] = False
    for i in range(1,n+1):
        if(is_prime[i]):
            lst.append(i)
    return lst

def binary_search(arr,l,r,x):           #binary Search
    if l<=r:
        mid=(l+r)//2
        if arr[mid]==x:
            return mid 
        elif arr[mid]>x:
            return binary_search(arr,l,mid-1,x)
        else:
            return binary_search(arr,mid+1,r,x)
    else:
        return -1

def find(x , parent):                                 #disjoint set Union FIND
    if parent[x]==x:
        return x 
    parent[x]=find(parent[x])
    return parent[x]

def union(x, y, parent):                              #disjoint set Union UNION
    p_x=find(x, parent)                               #find parents and if they are different make one their parent
    p_y=find(y, parent)                               # not considering rank yet
    if p_x != p_y:
        parent[p_y]=p_x
    
    
 
def MST(V, E, edge):                    # Kruskal's Algorithm edge is a list of edges in the format [ node 1 , node 2 , weight ]
    edge.sort(key = lambda x: x[2])     # sorting according to weights
    Included=[False]*(V+1)              # 0th index not considered
    Parent=[int(i) for i in range(V+1)]
    cost = 0
    count = 0
    for i in range(E):
        if find(edge[i][0], Parent)!=find(edge[i][1], Parent):
            cost+=edge[i][2]
            union(edge[i][0], edge[i][1], Parent)
            count+=1
        if count==V-1:
            break

    if count==V-1:
        return cost
    else:
        return -1

V,E=[int(i) for i in input().split()]

Edges=[]

for i in range(E):
    Edges.append([int(i) for i in input().split()])

print(MST(V,E,Edges))
