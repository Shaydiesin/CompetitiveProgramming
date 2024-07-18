N = 25643 

arr = [	0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657]


for i in range(len(arr)):
    for j in range(len(arr)):
        for k in range(len(arr)):
            # if arr[i] + arr[j] +arr[k] == N:
                print(arr[i],arr[j],arr[k], arr[i] + arr[j] +arr[k])