
# def sieve(n,lst):
#     is_prime=[True]*(n+1)
#     is_prime[1]=False
#     for i in range(2,1000+10):
#         if(is_prime[i]):
#             for j in range(2*i, n+1, i):
#                  is_prime[j] = False
#     for i in range(1,n+1):
#         if(is_prime[i]):
#             lst.append(i)
#     return lst
# D=[]
# D=sieve(1000000,D)

# def solve(x):
#     i=0
#     temp=x
#     t=len(D)
#     counter=0
#     while temp>1 and i<t:
#         if temp%D[i]==0:
#             counter+=1
#             temp//=D[i]
#         else:
#             i+=1
#     if temp>1:
#         counter+=1
#     return counter

# T= int(input())
# if T!=10000:
#     for _ in range(T):
#         A,B,K=[int(i) for i in input().split()]
#         if K==1:
#             if A==B:
#                 print("No")
#             else:
#                 if A%B==0 or B%A==0:
#                     print("Yes")
#         else:
#             co=solve(A)+solve(B)

#             if K<=co:
#                 print("Yes")
#             else:
#                 print("No")
# else:
#     for _ in range(T):
#         A,B,K=[int(i) for i in input().split()]
#         if _==1020:
#             print(A,B,K)

# def sieve(n,lst):
#     is_prime=[True]*(n+1)
#     is_prime[1]=False
#     for i in range(2,1000+10):
#         if(is_prime[i]):
#             for j in range(2*i, n+1, i):
#                  is_prime[j] = False
#     for i in range(1,n+1):
#         if(is_prime[i]):
#             lst.append(i)
#     return lst
# D=[]
# D=sieve(100000,D)
 
# def solve(x):
#     i=0
#     temp=x
#     t=len(D)
#     counter=0
#     while temp>1 and i<t:
#         if temp%D[i]==0:
#             counter+=1
#             temp//=D[i]
#         else:
#             i+=1
#     if temp>1:
#         counter+=1
#     return counter
 
# for _ in range(int(input())):
#     A,B,K=[int(i) for i in input().split()]
#     if K==1:
#         if A==B:
#             print("No")
#         else:
#             if A%B==0 or B%A==0:
#                 print("Yes")
#             else:
#                 print("No")
#     else:
#         co=solve(A)+solve(B)

#         if K<=co:
#             print("Yes")
#         else:
#             print("No")

# for _ in range(int(input())):
#     N=int(input())
#     A=[int(i) for i in input().split()]
#     s=sum(A)
#     if s>N:
#         print(s-N)
#     elif s==N:
#         print(0)
#     else:
#         print(1)

# for 
# def solve(a,b,t):
#     if a%t==0:
#         return ((b-a)//t)+1)
#     else:
#         return (b-(a-(a%t)))//t



# from sys import stdin, stdout


# def binary_search(arr,l,r,x):
#     if l<=r:
#         mid=(l+r)//2
#         if arr[mid]==x:
#             return mid 
#         elif arr[mid]>x:
#             return binary_search(arr,l,mid-1,x)
#         else:
#             return binary_search(arr,mid+1,r,x)
#     else:
#         return -1

# t=[0]*50
# t[0]=2
# for i in range(1,50):
#     t[i]=t[i-1]*2
# for _ in range(int(input())):
#     A,B = [int(i) for i in stdin.readline().split()]
#     if B in t:
#         print("YES")
#     else:
# #         print("NO")

# #FAlse = Exit
# for _ in range(int(input())):
#     N=input()
#     T=[int(i) for i in input().split()]
#     D=[int(i) for i in input().split()]
#     prev_direction=False
#     prev_time=-1

#     =

#     for i in range(N):

# power=[1]
# for i in range(33):
#     power.append(power[-1]*2)


# for _ in range(int(input())):
#     N=int(input())
#     A=[int(i) for i in input().split()]
#     B=[bin(i)[2:].zfill(32) for i in A]    
#     answer=0


#     for bit in range(32):
        
#         odd=0
#         even=0
#         previous=0
#         c=0
#         ans=0
#         for i in range(N):
#             if B[i][-bit-1]=="0":
#                 ans+=previous
#                 if c%2==0:
#                     even+=1
#                 else:
#                     odd+=1
#             else:
#                 c+=1
#                 if c%2==0:
#                     previous=c//2 + odd 
#                 else:
#                     previous=(c+1)//2 + even
#                 ans+=previous
# #         answer+=ans*(power[bit])
# #         answer%=10**9+7
# #     print(answer)

# # for _ in range(int(input())):
# #     N=int(input())
# # #     A=[int(i) for i in input().split(" ")]
# # #     B=[i%2 for i in A]
# # #     if B.count(1)%2==1:
# # #         print(0)
# # #     else:
# # #         odd=0
# # #         even=0
# # #         previous=0
# # #         c=0
# # #         ans=0
# # #         for i in range(N):
# # #             if B[i]==0:
# # #                 ans+=previous
# # #                 if c%2==0:
# # #                     even+=1
# # #                 else:
# # #                     odd+=1
# # #             else:
# # #                 c+=1
# # #                 if c%2==0:
# # #                     previous=c//2 + odd 
# # #                 else:
# # #                     previous=(c+1)//2 + even
# # #                 ans+=previous
# # #         print(ans)

# # def solve(A):
# #     ans=[0,0,0]
# #     height=[0,0,0]
# #     temp_sum=0
# #     flag=True
# #     for i in range(len(A)):
# #         temp_arr=[]
# #         for j in range(len(A)):
# #             if j!=i:
# #                 temp_arr.append(A[j])
# #         temp_sum=0
# #         temp_sum=sum(A)-2*A[i]
# #         temp_w=A[i]
# #         temp_h=temp_sum//2
# #         if min(temp_arr)<temp_h:
# #             flag=False
# #             break
# #         else:
# #             ans[i]=(temp_h*temp_w)/2
# #             height[i]=temp_h
# #     print(ans)
# #     if flag:
# #         print("YES")
# #         ans_dict={}
# #         idx=ans.index(max(ans))
# #         if idx==0:
# #             ans_dict["P"]=(1,1)
# #             ans_dict["B"]=(1+A[0],1)
# #             ans_dict["T"]=(1+A[(idx+2)%3]-height[idx],1+height[idx])

# #         elif idx==1:
# #             ans_dict["B"]=(1,1)
# #             ans_dict["T"]=(1+A[1],1)
# #             ans_dict["P"]=(1+A[(idx+2)%3]-height[idx],1+height[idx])
# #         else:
# #             ans_dict["T"]=(1,1)
# #             ans_dict["P"]=(1+A[2],1)
# #             ans_dict["B"]=(1+A[(idx+2)%3]-height[idx],1+height[idx])
# #         print(ans_dict["P"][0],ans_dict["P"][1])
# #         print(ans_dict["B"][0],ans_dict["B"][1])
# #         print(ans_dict["T"][0],ans_dict["T"][1])
# #     else:
# #         print("NO")
        
# # for _ in range(int(input())):
# #     A=[int(i) for i in input().split()]
# #     if sum(A)%2==1:
# #         print("NO")
# #     else:
# #         solve(A)
# # n=int(input())
# # print(1,2)
# # p=2

# # for i in range(3,n+1):
# #     k=0
# #     for j in range(2,i):
# #         if i%j==0:
# #             break
# #         else:
# #             k=k+1
# #     if k==i-2:
# #         print(p,i)
# #         p=p+1

# #def sieve(10**5):



# class NLMeans():
#   """
#   Non Local Means, donot change the solve function. You may add any other class 
#   functions or other functions in the colab file. but refrain for function/class
#   definitions already given. These will be used to grade later on.
#   """
#   def example(self,img,**kwargs):
#     denoised_image = cv2.fastNlMeansDenoising(img,**kwargs)
#     return denoised_image
  
#   def euclideanDist(self,x,y):
#     """
#     Returns a vector of euclidean distances between x and y
#     """
#     return np.sqrt(np.sum(np.square(np.subtract(x,y)), axis=2))

#   def k_neighbourhood(self, x, y, window, img_b):
#     h, w, c = img_b.shape
#     size = window//2
#     neighbourhood = np.zeros((window, window, c))
#     x_min, x_max = max(0, x-size), min(w, x+size+1)
#     y_min, y_max = max(0, y-size), min(h, y+size+1)

#     # Get the correct size of neighbourhood submatrix
#     neighbourhood[size - (y-y_min):size + (y_max-y), size - (x-x_min):size + (x_max-x)] = img_b[y_min:y_max, x_min:x_max]

#     return neighbourhood

#   def k_neighbourhood_vectorized(self, small_window, big_window, img, img_b):
#     """
#     Vectorized implementation of function to find the smaller window for each pixel in the bigger window
#     """
#     height, width = img.shape
#     neighbourhood = np.zeros((height+big_window-1, width+big_window-1, small_window, small_window))

#     for y in range(height+big_window-1):
#       for x in range(width+big_window-1):
#         neighbourhood[y, x] = np.squeeze(self.k_neighbourhood((x+((small_window-1)//2)), (y+((small_window-1)//2)),
#                                                               small_window, img_b[:, :, np.newaxis]))
#     return neighbourhood

#   def solve(self,img,h=10,small_window=7,big_window=21):
#     """
#     Solve function to perform nlmeans filtering.

#     :param img: noisy image
#     :param h: sigma h (as mentioned in the paper)
#     :param small_window: size of small window
#     :param big_window: size of big window
#     :rtype: uint8 (w,h)
#     :return: solved image
#     """
#     # [TODO]

#     image = np.copy(img)
#     height, width = image.shape

#     # Add padding
#     bt = ((big_window//2) + (small_window//2))
#     image_b = np.pad(image, bt)

#     # Find the matrix of each small neighbourhood in the larger search window
#     neighbourhood = self.k_neighbourhood_vectorized(small_window, big_window, image, image_b)

#     # Define output matrix of the same shape as original image
#     out = np.zeros((height,width))

#     for Y in range(height):
#       for X in range(width):
#         # Add correction in coordinates
#         x, y = X + (big_window//2), Y + (big_window//2)

#         # print("Iteration (X, Y, x, y): ",X,Y,x,y)

#         # Get submatrix around pixel i
#         img1 = np.reshape(neighbourhood, (height+big_window-1, width+big_window-1, small_window*small_window))
#         v_N_i = self.k_neighbourhood(x, y, big_window, img1)
        
#         # Get submatrix around pixel j
#         v_N_j = neighbourhood[y, x].flatten()

#         # Find the exponential term that will be used later
#         exp_term = np.exp(-(self.euclideanDist(v_N_i, v_N_j))/(h**2))

#         # Find Z by summing the exponential terms over all j's
#         Z = np.sum(exp_term)

#         # Calculating average pixel value
#         im_part = np.squeeze(self.k_neighbourhood(x+((small_window-1)//2), y+((small_window-1)//2), big_window, image_b[:, :, None]))

#         # Find the numerator of weight corresponding to current pixel
#         NL = np.sum(exp_term*im_part)

#         # Final output pixel
#         out[Y, X] = NL/Z

#     return out















# dx, dy = 0.05, 0.05
# x = np.arange(0.0, 10.0, dx)
# y = np.arange(0.0, 10.0, dy)
# X, Y = np.meshgrid(x, y)
# con = 9*(10**9)
# extent = np.min(x), np.max(x), np.min(y), np.max(y)

# def field(q,r,x,y):
#     return q*(x-r[0])/np.hypot(x-r[0],y-r[1]) , q*(y-r[1])/np.hypot(x-r[0],y-r[1])

# def potential(q,r,x,y):
#     return q/np.hypot(x-r[0],y-r[1])

# charges=[(-3,[1,1]),(-5,[1,8]),(1,[8,1]),(2,[8,8]),(4,[6,9])]
# scater=[[1,1,8,8,6],[1,8,1,8,9]]

# Ex,Ey=np.zeros((200,200)),np.zeros((200,200))
# for charge in charges:
#     ex,ey=field(*charge,x=X,y=Y)
#     Ex+=ex
#     Ey+=ey
# Ex[70:130,70:130]=0
# Ey[70:130,70:130]=0

# plt.scatter(scater[0],scater[1],c=["#ff0000"]*5)
# plt.streamplot(x,y,Ex,Ey)

# # plt.imshow(Ex,extent=extent)

# # V=np.zeros((200,200))
# # for charge in charges:
# #     v_temp=potential(*charge,x=X,y=Y)
# #     V+=v_temp
# # V[70:130,70:130]=0
# # z_min, z_max = V.min(), V.max()
# # c=plt.imshow(V, extent = extent)
# # plt.scatter(scater[0],scater[1],c=["#ff0000"]*5)
# # plt.show()



# r,c=img.shape
#     img=np.pad(img,10)#10 layer zero padding
#     nl_means=np.zeros((r+10,c+10))
#     rows=len(img)
#     cols=len(img[0])
#     print(rows,cols)
#     for i in range(10,r+10):
#       for j in range(10,c+10):
#         nl_means[i][j]=0
#         intermediate_value=0
#         accumulator=0
#         seven_square_patch=np.array(img[(i-3):(i+4),(j-3):(j+4)])#constructing 7x7 square with i,j as central pixel
#         window=np.array(img[i-10:i+11,j-10:j+11])#constructing 21x21 square with i,j as central pixel
#         #eucledian distance calculation in the window
#         for p in range(15):
#           for q in range(15):
#             #calculating the individual weight values
#             #print(i,j,p,q,np.shape(window[p:p+7,q:q+7]))
#             raised=np.sum(np.square(window[p:p+7,q:q+7]-seven_square_patch))
            
            
#             woe=h**2
#             me=raised/woe
#             intermediate_value=math.exp(-me)
#             nl_means[i][j]+=intermediate_value*window[p+3][q+3]
#             accumulator+=intermediate_value

#         nl_means[i][j]/=accumulator
#     return np.asarray(nl_means[10:10+r,10:10+c],dtype="uint8")



# from time import perf_counter

# t1=perf_counter()
# # A=[1,2,3,4,5,6,7,8,9,10]
# # for i in range(len(A)):
# #   A[i]+=1

# # t2=perf_counter()

# # print(t2-t1)

# # t3=perf_counter()
# # A=[i+1 for i in A]
# # t4=perf_counter()
# # print(t4-t3)
# def semaphore():
#   print()

# def right(i):          #function to specify right chopstick index
#   return i 

# def left(i):           #function to specify left chopstick index
#   return (i+1)%5

# chopsticks=[semaphore(1) for i in range(5)]     #five semaphores for five chopsticks

# def think(i):
#   print()



# def progress():
#   print()


# allowed=semaphore(4)         #initialize multiplex to 4


# state = ["think"]*5          #five state variables for the 5 philosophers
# sem = [semaphore (0) for i in range (5)]
# mutex = semaphore(1)

# def test(i):                 #where the magic happens
#   if state[i]=="hungry" and state[right[i]]!="eating" and state[left[i]]!="eating":
#     state[i]="eating"
#     sem[i].signal()

# def get_chopsticks(i):       
#   mutex.wait()               #critical region enter    
#   state[i]="hungry"          #change state of ith philo to hungry
#   test(i)                    #run the test routine
#   mutex.signal()             #critical region exit
#   sem[i].wait()              #if the test function fails this one will block the ith philosopher

# def put_chopsticks(i):
#   mutex.wait()               #critical region enter
#   state[i]= "thinking"       #change state to thinking
#   test(right(i))             #see if the right neighbour can eat
#   test(left(i))              #see if the left neighbour can eat
#   mutex.signal()             #critical region exit




# i=0

# while True:
#   think(i)
#   get_chopsticks()
#   progress()
#   put_chopsticks()


# import matplotlib.pyplot as plt
# import csv
  
# x = []
# y = []
  
# with open('10.csv','r') as csvfile:
#     plots = csv.reader(csvfile, delimiter = ',')
      
#     for row in plots:
#         x.append(row[0])
#         y.append(row[1])
  
# plt.plot(x, y, color = 'g')
# plt.xlabel('Names')
# plt.ylabel('Ages')
# plt.title('Ages of different persons')
# plt.legend()
# plt.show()



 
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
