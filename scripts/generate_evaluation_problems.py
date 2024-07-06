import random

def generate_evaluation_problems():
    problems = []

    # Simple arithmetic problems
    for _ in range(10):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        problems.append(f"What is {a} + {b}?")

    # Algebra problems
    for _ in range(10):
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        problems.append(f"Solve for x: {a}x + {b} = 0")

    # Calculus problems
    problems.append("Differentiate the function: f(x) = x^2 + 3x + 2")
    problems.append("Integrate the function: f(x) = 2x + 1")

    # Complex problem-solving tasks
    problems.append("Solve the system of equations: 2x + 3y = 5, 4x - y = 3")
    problems.append("Find the roots of the quadratic equation: x^2 - 4x + 4 = 0")

    return problems

if __name__ == "__main__":
    evaluation_problems = generate_evaluation_problems()
    with open("/home/ubuntu/chat-agent/VishwamAI-main/scripts/evaluation_problems.txt", "w") as file:
        for problem in evaluation_problems:
            file.write(problem + "\n")
