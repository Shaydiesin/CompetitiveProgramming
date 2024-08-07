import sys
from collections import deque
import heapq

# Setting high recursion depth for dfs
sys.setrecursionlimit(250000) 

# def inp():
#     return(int(input()))
# def inlt():
#     return(list(map(int,input().split())))
# def insr():
#     return(input().strip())
# def invr():
#     return(map(int,input().split()))

# sys.stdout.write(str(ans)+"\n")

# N= 100

#
#                        [0, n-1]
#                       /       \
#                 [0, n/2]      [n/2+1,n-1]
#                   /     \
#              [0,n/4]  [n/4, n/2]  
#               /
#             ...              
#             /
#       [0,0]             

# Binary Exponentiation for a^b modulo m
def binExp(a,b,mod):
    ans = 1
    while b>0:
        if b&1:
            ans*=a 
            ans%=mod
        a = (a*a)%mod 
        b = b>>1
        
    return ans


def segmentTree(arr,node,l,r):         #Segment tree build 
    if l==r:
        segment_tree[node] = arr[l]
        return segment_tree[node]
    else:
        mid = (l+r)//2
        left = segmentTree(arr,2*node+1,l,mid)
        right = segmentTree(arr,2*node +2, mid+1,r)
        segment_tree[node] = left+right
        return segment_tree[node]

arr = list(range(1,6))
segment_tree = [-1]*(4*len(arr))
segmentTree(arr,0,0,len(arr)-1)

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


def binary_search(arr,l,r,x,idx):           #binary Search l = 0, r =len(arr - 1)
    if l<=r:
        mid=(l+r)//2
        if arr[mid]==x:
            idx = mid
            return binary_search(arr,l,mid-1,x,idx) 
        elif arr[mid]>x:
            return binary_search(arr,l,mid-1,x,idx)
        else:
            return binary_search(arr,mid+1,r,x,idx)
    else:
        return idx

def geq_bin_search(arr,l,r,x,idx):          #binary Search l = 0, r =len(arr - 1) To find element just greater than equal to x
    if l<=r:
        mid=(l+r)//2
        if arr[mid]>=x:
            idx = mid
            return geq_bin_search(arr,l,mid-1,x,idx) 
        elif arr[mid]<x:
            return geq_bin_search(arr,mid+1,r,x,idx)
    else:
        return idx

def leq_bin_search(arr,l,r,x,idx):          #binary Search l = 0, r =len(arr - 1) To find element just lesss than equal to x
    if l<=r:
        mid=(l+r)//2
        if arr[mid]==x:
            idx = mid                       #comment this line for strictly greater
            return leq_bin_search(arr,l,mid-1,x,idx)
        elif arr[mid]<x:
            idx = mid
            return leq_bin_search(arr,mid+1,r,x,idx)
        elif arr[mid]>x:
            return leq_bin_search(arr,l,mid - 1,x,idx)
    else:
        if idx!=-1:
            return binary_search(arr,0,idx,arr[idx],idx)
        return idx



#   [1,2,3,4,5,6,7,8,9,10]
#   [ [1,3], 2,4, [5,6] ,7,8,9,10 ]

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

def dfs(node):
    adj ={}
    visited = [False] #*N
    print(node)
    visited[node] = True                           #1 indexing 

    for i in adj[node]:
        if not visited[i]:
            visited[i] = True
            dfs(i)



def bfs(node,adj):
    N = len(adj)
    visited = [False]*N 
    visited[node] = True 

    q = deque()

    q.append(node)             # update the type for node
    while(len(q)!=0):
        curr = q.popleft()
        for i in adj[curr]:
            if not visited[i]:
                visited[i] = True 
                print(q)
                q.append(i)



class DijkstraNode:
    def __init__(self, node, value):
        self.Node = node
        self.Value = value
    
    def __lt__(self, other):
        return self.Value < other.Value


# The general idea is to start at the source Node
# When you relaxed a vertex add it to the priority queue
# using the above data type made

def Dijkstra(node,adj):
    N = len(adj)
    dist = [10**9]*N               # increase the initial value even further if requireds
    dist[node] = 0

    pq = []

    tmp = DijkstraNode(node,0)

    heapq.heapify(pq)
    heapq.heappush(pq,tmp)
    
    while(len(pq)!=0):
        curr = heapq.heappop(pq)
        for i in adj[curr.Node]:
            if curr.dist + i[1] < dist[i[0]]:
                dist[i[0]] = curr.dist + i[1]
                tmp = DijkstraNode(i[0],dist[i[0]])
                heapq.heappush(pq,tmp)
    
    return dist



# In the given edge list Keep relaxing N-1 times (N) is the number of nodes
# If in the Nth iteration there is improvement it means there is negative cycle
def BellManFord(edges,N):
    dist = [10**9]*N 
    
    return 


# Given the graph fix a vertex "K" and using it relax between every other 
# vertex pair i,j 
def FloydWarshall(adjMat):
    
    return 




# for dp on trees and tree traversals
# def dfs_tree(node,parent):
#     visited[node]= True

#     g_st[node]= A[node-1]
#     for child in adj[node]:
#         if child!=parent:
#             if not visited[child]:
#                 if g_st[child]!=0:
#                     g_st[node]=gcd(g_st[node],g_st[child])
#                 else:
#                     g_st[node]=gcd(g_st[node],dfs(child,node))
# 
#     return g_st[node]

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

# V,E=[int(i) for i in input().split()]

# Edges=[]

# for i in range(E):
#     Edges.append([int(i) for i in input().split()])

# print(MST(V,E,Edges))

A = [1 ,3, 4, 5]

print(binary_search(A,0,len(A)-1,int(input())))