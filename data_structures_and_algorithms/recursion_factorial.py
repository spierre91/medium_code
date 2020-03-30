def recursive_factorial(n):
    if n == 1:
        return 1
    else:
        return n*recursive_factorial(n-1)
print("4! =", recursive_factorial(4))

print("6! =", recursive_factorial(6))
