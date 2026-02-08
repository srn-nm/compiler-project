def compute_fibonacci(terms):
    """
    Compute Fibonacci numbers for given terms count
    """
    if terms <= 0:
        return []
    if terms == 1:
        return [0]
    if terms == 2:
        return [0, 1]
    
    result = [0, 1]
    for idx in range(2, terms):
        new_value = result[idx-1] + result[idx-2]
        result.append(new_value)
    
    return result

def get_prime_numbers(max_value):
    # Generate prime numbers below max_value
    prime_list = []
    for current in range(2, max_value + 1):
        prime_flag = True
        for divisor in range(2, int(current**0.5) + 1):
            if current % divisor == 0:
                prime_flag = False
                break
        if prime_flag:
            prime_list.append(current)
    return prime_list

def bubble_sort(arr):
    """Sort array using bubble sort algorithm"""
    size = len(arr)
    swapped = False
    for i in range(size):
        swapped = False
        for j in range(0, size-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

class Person:
    def __init__(self, full_name, years_old):
        self.full_name = full_name
        self.years_old = years_old
    
    def get_name(self):
        return self.full_name
    
    def get_age(self):
        return self.years_old

class UniversityStudent(Person):
    def __init__(self, name, age, scores):
        super().__init__(name, age)
        self.scores = scores
    
    def compute_average_score(self):
        if len(self.scores) == 0:
            return 0.0
        total = 0.0
        for score in self.scores:
            total += score
        return total / len(self.scores)
    
    def show_details(self):
        average = self.compute_average_score()
        print(f"Name: {self.full_name}")
        print(f"Age: {self.years_old}")
        print(f"Average Score: {average:.1f}")

# Driver code
if __name__ == "__main__":
    # Fibonacci calculation
    fib_output = compute_fibonacci(8)
    print(f"Fibonacci sequence (8 terms): {fib_output}")
    
    # Prime numbers
    primes = get_prime_numbers(30)
    print(f"Prime numbers up to 30: {primes}")
    
    # Sorting example
    data = [45, 23, 67, 12, 89, 34, 56]
    sorted_data = bubble_sort(data.copy())
    print(f"Original: {data}")
    print(f"Sorted: {sorted_data}")
    
    # Student information
    grades = [88, 92, 76, 95, 84]
    student = UniversityStudent("Bob", 22, grades)
    student.show_details()
    
    # Additional test
    x = 10
    y = 20
    sum_result = x + y
    print(f"Sum of {x} and {y} is {sum_result}")