def calculate_fibonacci(n):
    """
    Calculate Fibonacci sequence up to n terms
    """
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    fib_sequence = [0, 1]
    for i in range(2, n):
        next_term = fib_sequence[i-1] + fib_sequence[i-2]
        fib_sequence.append(next_term)
    
    return fib_sequence

def find_primes(limit):
    # Find all prime numbers up to limit
    primes = []
    for num in range(2, limit + 1):
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes

def sort_list(numbers):
    """Sort a list of numbers in ascending order"""
    n = len(numbers)
    for i in range(n):
        for j in range(0, n-i-1):
            if numbers[j] > numbers[j+1]:
                numbers[j], numbers[j+1] = numbers[j+1], numbers[j]
    return numbers

class Student:
    def __init__(self, name, age, grades):
        self.name = name
        self.age = age
        self.grades = grades
    
    def calculate_average(self):
        if not self.grades:
            return 0.0
        return sum(self.grades) / len(self.grades)
    
    def display_info(self):
        avg = self.calculate_average()
        print(f"Student: {self.name}")
        print(f"Age: {self.age}")
        print(f"Average Grade: {avg:.2f}")

# Main execution
if __name__ == "__main__":
    # Test fibonacci
    fib_result = calculate_fibonacci(10)
    print(f"First 10 Fibonacci numbers: {fib_result}")
    
    # Test primes
    prime_result = find_primes(50)
    print(f"Primes up to 50: {prime_result}")
    
    # Test sorting
    numbers_to_sort = [64, 34, 25, 12, 22, 11, 90]
    sorted_numbers = sort_list(numbers_to_sort.copy())
    print(f"Sorted list: {sorted_numbers}")
    
    # Test student class
    student1 = Student("Alice", 20, [85, 90, 78, 92, 88])
    student1.display_info()