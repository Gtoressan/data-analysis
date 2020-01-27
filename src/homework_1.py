import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Python/task 1.
# Enter numbers a, d и N. Find sum of the first N members of arithmetic progression with the first member a and
# difference d, without using formula for the sum.
def task_1():
    print('Enter "a":', end=' ')
    a = int(input())

    print('Enter "d":', end=' ')
    d = int(input())

    print('Enter "N":', end=' ')
    n = int(input())

    for _ in range(n - 1):
        a += d

    print(f"Your arithmetic regression is {a}\n")


# Python/task 2.
# Enter a number N. Find sum of the first N members of harmonic series.
def task_2():
    print('Enter "N":', end=' ')
    n = int(input())
    answer = 0.0

    for i in range(n):
        answer += 1.0 / (i + 1)

    print(f"Your harmonic series sum is {answer}\n")


# Python/task 3.
# Enter an integer number N. Check if it is a prime number.
def task_3():
    print('Enter "N":', end=' ')
    n = int(input())

    if n > 1:
        for i in range(2, int(n / 2) + 1):
            if n % i == 0:
                print(f'The number {n} is consistence!\n', n)
                return

        print(f'The number {n} is prime!\n', n)


# Python/task 4.
# Enter the first N prime numbers.
def task_4():
    print('Enter "N":', end=' ')
    n = int(input())
    prime = 2

    while n >= 1:
        print(prime, end=' ')
        prime += 1

        while not isPrime(prime):
            prime += 1

        n -= 1
    print()


def isPrime(number):
    if number > 1:
        for i in range(2, int(number / 2) + 1):
            if number % i == 0:
                return False

        return True
    return False


# Python/task 5.
# There are numbers: a, b, c. Without using functions min, max and other functions, find the maximum number.
def task_5():
    print('Enter "a":', end=' ')
    a = int(input())

    print('Enter "b":', end=' ')
    b = int(input())

    print('Enter "c":', end=' ')
    c = int(input())

    if a < b:
        a = b

    if a < c:
        print(f'The maximum is {c}\n')
    else:
        print(f'The maximum is {a}\n')


# Numpy/task 1.
# Create two random arrays 'a' and 'b' with the same length.
# Calculate the following distances between the arrays:
# 1. Euclidean Distance.
# 2. Manhattan Distance.
# 3. Cosine Distance.
def task_6():
    print('Enter the length of arrays:', end=' ')
    length = int(input())

    a = np.random.rand(length)
    b = np.random.rand(length)
    euclidean_distance = np.linalg.norm(a - b)
    manhattan_distance = (a - b).sum()
    cosine_distance = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    print(f'Euclidean Distance is {euclidean_distance}')
    print(f'Manhattan Distance is {manhattan_distance}')
    print(f'Cosine Distance is {cosine_distance}\n')


# Numpy/task 2.
# Create a random array with length of 10 and with sum of its elements equals to 2.
def task_7():
    a = np.random.rand(10)
    a /= a.sum() / 2
    print(f'The sum of array elements is {a.sum()}')
    print(a)
    switcher = input()


# Numpy/task 3.
# Create a random array with length of 100. Transform the array so, that:
# 1. Maximum element(s) value is 1.
# 2. Minimum element(s) value is 0
# 3. Other values are in interval 0-1 with keeping the order.
def task_8():
    a = list(np.random.rand(100))
    a[a.index(max(a))] = 1
    a[a.index(min(a))] = 0
    print(f'Maximum value is {max(a)}')
    print(f'Minimum value is {min(a)}')
    print(*a)
    switcher = input()


# Numpy/task 4.
# Create a random array with length of 20 with integers from [0,50]. Select elements that are smaller than 5th
# percentile and larger than 95th percentile
def task_9():
    a = np.random.randint(0, 50, size=20)
    per5 = np.percentile(a, 5)
    per95 = np.percentile(a, 95)
    print(f'Elements smaller than 5th percentile {per5}:', a[a < per5])
    print(f'Elements greeter than 95th percentile {per95}:', a[a > per95])
    print(*a)
    switcher = input()


# Numpy/task 5.
# Create an array with shape of {5; 6} with integer from 0 to 50. Print a column that contains the maximum
# element of the array.
def task_10():
    a = np.random.randint(0, 50, size=[5, 6])
    print(a)
    print('Your column is:', a[:, a.argmax() % 6])
    switcher = input()


# Numpy/task 6.
# Replace all missing values in the following array with mean
def task_11():
    arr = np.random.rand(10)
    idx = np.random.randint(0, 10, 4)
    arr[idx] = np.nan

    print('Beginning array:', *arr)
    arr_mean = arr[~np.isnan(arr)].mean()
    arr[np.isnan(arr)] = arr_mean
    print('Mean is', arr_mean)
    print('New array:', *arr)
    print()
    switcher = input()


# Numpy/task 7.
# Download file 1 and file 2 into a directory with this notebook. Using function loadtxt in numpy load data from the
# first file. Assign y = D[:,0] аnd X = D[:, 1:].
# We will use one magic formula to find linear regression coefficients. You will prove this formula on your next
# lectures of the course.
def task_12():
    D = np.loadtxt('../templates/tutorial_dataset_2.csv', skiprows=1, delimiter=',')
    y = D[:, 0]
    X = D[:, 1:]
    b = np.linalg.inv(X.T @ X) @ X.T @ y
    new_y = X @ b
    res = y - new_y

    plt.figure(figsize=(9, 6))
    plt.scatter(new_y, res)
    plt.show()

    switcher = input()


if __name__ == '__main__':
    while True:
        task_12()
