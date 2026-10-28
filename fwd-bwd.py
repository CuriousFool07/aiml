
# Forward and Backward Chaining

facts = {"fever", "cough"}

rules = [
    ({"fever", "cough"}, "flu"),
    ({"flu"}, "needs_rest"),
    ({"needs_rest"}, "stay_home")
]

# Forward Chaining
def forward_chaining():
    known = set(facts)
    changed = True

    while changed:
        changed = False

        for conditions, result in rules:
            if conditions.issubset(known) and result not in known:
                known.add(result)
                changed = True

    return known

# Backward Chaining
def backward_chaining(goal, known):
    if goal in known:
        return True

    for conditions, result in rules:
        if result == goal:
            if all(backward_chaining(c, known) for c in conditions):
                return True

    return False

# Display results
print("Initial Facts:", facts)

print("\nForward Chaining:")
print("Derived Facts:", forward_chaining())

goal = "flu"
print("\nBackward Chaining:")
print("Goal:", goal)
print("Can goal be proved?", backward_chaining(goal, facts))
