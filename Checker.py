import numpy as np
import math

# Parse driving times
driving_time = np.loadtxt("drivingtimes.txt", dtype=int)

# Parse orders
with open("orders.txt") as f:
    lines = f.readlines()[1:]  # Skip the first line with the header

profit = np.zeros(1177, dtype=int)
duration = np.zeros(1177, dtype=int)
nodeID = np.zeros(1177, dtype=int)
orderID = np.zeros(1177, dtype=int)
allowed = np.zeros((1177, 20), dtype=bool)
order_done = np.zeros(1177, dtype=bool)
order_lookup = {}

for i, line in enumerate(lines):
    order_data = line.strip().split("\t")
    orderID[i] = int(order_data[0])
    nodeID[i] = int(order_data[2])
    duration[i] = int(order_data[5]) * 60  # Multiply by 60 to convert to seconds just like driving times
    profit[i] = int(order_data[6]) * 60  # Also multiplied by 60, this way one unit of profit is equal to one unit (second) of time
    
    order_lookup[orderID[i]] = i
    
    # Parse what employees can do what
    for j in range(20):
        if order_data[7 + j] == "1":
            allowed[i][j] = True

# Read & check the solution file
with open("output.txt") as f:
    lines = f.readlines()

claimed_profit = int(lines[0])  # First line is (supposed) profit of solution
real_profit = 0
valid = True

pos = 1

for i in range(20):
    tasks_employee = int(lines[pos])
    pos += 1
    
    travel_time = 0
    order_profit = 0
    working_time = 0
    
    cur = 251  # Starting location
    
    for _ in range(tasks_employee):
        order = int(lines[pos])
        pos += 1
    
        idx = order_lookup[order]
        
        # Each order may be performed at most once
        if order_done[idx]:
            print(f"ERROR: order is performed multiple times: {order}")
            valid = False
        order_done[idx] = True
        
        # Check that employee is authorized
        if not allowed[idx][i]:
            print(f"ERROR: order {order} is performed by incapable employee: {i + 1}")
            valid = False
        
        # Accounting
        order_profit += profit[idx]
        working_time += duration[idx]
        
        # Figure out time spent driving
        travel_time += driving_time[cur, nodeID[idx]]
        
        cur = nodeID[idx]
    
    travel_time += driving_time[cur, 251]  # Back home
    
    if travel_time + working_time > 8 * 60 * 60:
        print(f"ERROR: employee {i + 1} works overtime: {travel_time + working_time}s")
        valid = False
    
    print(f"Employee {i + 1} - Profit: {order_profit} - Driving time: {travel_time} - Working time: {working_time}")
    
    real_profit += order_profit - travel_time - working_time

if real_profit // 60 != claimed_profit and real_profit // 60 + 1 != claimed_profit:
    print(f"ERROR: claimed profit of {claimed_profit} does not match actual value achieved of {real_profit // 60}")
elif not valid:
    print(f"ERROR: solution is invalid but profit of {claimed_profit} does match")
else:
    print(f"CORRECT: solution meets all constraints and has profit of {round(real_profit * 100 / 60) / 100.0}")

if claimed_profit < 0:
    print(f"WARNING: you should aim for positive profit, value of {claimed_profit} is negative")
