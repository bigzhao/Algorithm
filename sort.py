import random
import time
import numpy as np


def bubble_sort(unsort_list):
    length = len(unsort_list)
    for i in range(length):
        mark = 0
        for j in range(1, length - i):
            if unsort_list[j] < unsort_list[j - 1]:
                unsort_list[j], unsort_list[j - 1] = unsort_list[j - 1], unsort_list[j]
                mark = 1
        if not mark:
            return unsort_list
    return unsort_list


# print bubble_sort([1,2,3,4])
def select_sort(unsort_list):
    length = len(unsort_list)
    for i in range(length):
        rank = i
        for j in range(i + 1, length):
            if unsort_list[rank] > unsort_list[j]:
                rank = j
        unsort_list[rank], unsort_list[i] = unsort_list[i], unsort_list[rank]
    return unsort_list


# print select_sort([1,3,8,7,5,6])
def partition(A, p, q):
    key = A[p]
    i, j = p, p + 1
    while j <= q:
        # print i, j
        if A[j] < key:
            A[j], A[i + 1] = A[i + 1], A[j]
            i += 1
        j += 1
    A[p], A[i] = A[i], A[p]
    return i


def quick_sort(unsort_list, head, tail):
    if head >= tail:
        return
    r = partition(unsort_list, head, tail)
    quick_sort(unsort_list, head, r - 1)
    quick_sort(unsort_list, r + 1, tail)
    time.sleep(0.1)
    return


def merge(arr1, arr2):
    arr3 = []
    i, j = 0, 0
    while i < len(arr1) and j < len(arr2):
        # print i, j
        if arr1[i] < arr2[j]:
            arr3.append(arr1[i])
            i += 1
        else:
            arr3.append(arr2[j])
            j += 1
    if i < len(arr1):
        arr3 += arr1[i:]
    elif j < len(arr2):
        arr3 += arr2[j:]
    return arr3


def merge_sort(arr):
    if len(arr) <= 1:
        print arr
        return arr
    length = len(arr)
    arr1 = merge_sort(arr[: length / 2])
    arr2 = merge_sort(arr[length / 2:])
    # print arr1, arr2
    arr3 = merge(arr1, arr2)
    return arr3


def rand_partition(A, p, q):
    r = random.randint(p, q)
    A[p], A[r] = A[r], A[p]
    return partition(A, p, q)


def rand_quick_sort(A, p, q):
    if p >= q:
        return
    r = rand_partition(A, p, q)
    rand_quick_sort(A, p, r - 1)
    rand_quick_sort(A, r + 1, q)
    time.sleep(0.1)
    return


def double_partition(A, p, q):
    pivot = A[p]
    i = p
    j = q
    while i < j:
        while i < j and A[j] > pivot:
            j -= 1
        A[i] = A[j]
        while i < j and A[i] < pivot:
            i += 1
        A[j] = A[i]
    A[j] = pivot
    return i


def counting_sort(A, p, q):
    """number in A is 0-99"""
    B = [0 for _ in range(p, q + 1)]
    C = [0 for _ in range(100)]
    for i in A[p: q + 1]:
        C[i] += 1
    for i in range(1, len(C)):
        C[i] += C[i - 1]
    for i in A[::-1]:
        B[C[i] - 1] = i
        C[i] -= 1
    return B


def insertion_sort(A):
    length = len(A)
    # print A
    for i in range(1, length):
        j = i
        while j > 0 and A[j] < A[j - 1]:
            A[j], A[j - 1] = A[j - 1], A[j]
            j -= 1
            # print A


def bucket_sort(A):
    """number in A is 0-99"""
    buckets = [[] for _ in range(10)]
    for i in A:
        buckets[int(i / 10)].append(i)
    for bucket in buckets:
        insertion_sort(bucket)
    ret = []
    for i in buckets:
        ret += i
    return ret


def radix_sort(A):
    # use counting sort
    b = [0 for _ in range(len(A))]
    c = [0 for _ in range(10)]
    for i in A:
        c[i % 10] += 1
    for i in range(1, len(c)):
        c[i] += c[i - 1]
    for i in A[::-1]:
        b[c[i % 10] - 1] = i
        c[i % 10] -= 1
    A = b[:]
    b = [0 for _ in range(len(A))]
    c = [0 for _ in range(10)]
    for i in A:
        c[i / 10] += 1
    for i in range(1, len(c)):
        c[i] += c[i - 1]
    for i in A[::-1]:
        b[c[i / 10] - 1] = i
        c[i / 10] -= 1
    return b


def is_prime(n):
    for i in xrange(2, int(pow(n, 0.5))+1):
        if n % i == 0:
            return False
    return True


def prime_generator(l):
    i = 2
    for j in xrange(l):
        while 1:
            if is_prime(i):
                yield i
                i += 1
                break
            i += 1


def shell_sort(A):
    length = len(A)
    gap = length / 2
    while gap >= 1:
        for i in xrange(gap, length):
            j = i
            while j - gap >= 0 and A[j] < A[j - gap]:
                A[j], A[j - gap] = A[j - gap], A[j]
                j -= gap
        gap /= 2
    print A


if __name__ == "__main__":
    # A = 100 * np.random.rand(1000)
    # # B.reverse()
    # t = time.time()
    # rand_quick_sort(A, 0, len(A) - 1)
    # print ("random quick sort: ", time.time()-t)
    # t = time.time()
    # quick_sort(B, 0, len(A) - 1)
    # print time.time()-t
    # print merge_sort([5,7,9,15,20,4,1,2,60,44])
    # double_partition([6,5,4], 0, 2)
    # t = time.time()
    # A = bucket_sort(A)
    # print A
    # print "bucket sort time: ", time.time()-t

    # t = time.time()
    # counting_sort(A, 0, len(A) - 1)
    # print "bucket sort time: ", time.time()-t
    shell_sort([73, 22, 93, 43, 55, 14, 28, 65, 39, 81])
