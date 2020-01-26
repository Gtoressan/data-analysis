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


if __name__ == '__main__':
    while True:
        task_5()
