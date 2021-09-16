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
D=[]
D=sieve(1000000,D)

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

 
def MST(V, E, edge):                    #Kruskal's Algorithm edge is a list of edges in the format [ node 1 , node 2 , weight ]
    edge.sort(key = lambda x: x[2])     # sorting according to weights
    Included=[False]*(V+1)              #0th index not considered
    cost = 0
    count = 0
    for i in range(E):
        if not Included[edge[i][0]] or not Included[edge[i][1]]:
            cost+=edge[i][2]
            if not Included[edge[i][0]]:
                Included[edge[i][0]]=True
                count+=1
            if not Included[edge[i][1]]:
                Included[edge[i][1]]=True
                count+=1
            if count==V-1:
                break
    return cost

V,E=[int(i) for i in input().split()]

Edges=[]

for i in range(E):
    Edges.append([int(i) for i in input().split()])


#every weighted edge is considered as [ node 1 , node 2 , weight ]

print(MST(V, E ,Edges))
