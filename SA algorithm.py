import math
import random
import csv
import time
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

# Helper function to randomly select an element from a set without converting it to a list.
def random_choice_set(s):
    idx = random.randrange(len(s))
    for i, elem in enumerate(s):
        if i == idx:
            return elem

class SimAnneal:
    def __init__(self, orders, driving_times, headquarters=251):
        self.orders = {order['id']: order for order in orders}
        self.driving_times = driving_times
        self.headquarters = headquarters
        self.student_count = 20
        self.max_time = 480 * 60  # seconds
        self.cost_per_second = 1 / 60

        self.T = 12
        self.alpha = 0.999997
        self.stopping_T = 1e-8
        self.stopping_iter = 6000000
        self.iteration = 1

        # The current solution: one route per student.
        self.cur_solution = [[] for _ in range(self.student_count)]
        # Cache per-route fitness values (computed as: profit - time*cost - penalty)
        self.route_fitness = [0 for _ in range(self.student_count)]
        # New caches for route time and profit, so we can update them incrementally.
        self.route_time = [0 for _ in range(self.student_count)]
        self.route_profit = [0 for _ in range(self.student_count)]
        self.cur_fitness = None  # overall fitness (sum of route fitness)
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
        self.temperature_history = []

        self.operators = ['add', 'remove', '2-opt', 'swap', 'relocate']
        self.op_attempts = {op: 0 for op in self.operators}
        self.op_success = {op: 0 for op in self.operators}

    def evaluate_route(self, route):
        """
        Compute the route's total time, total profit, and its fitness component.
        Fitness component = profit - (time * cost_per_second) - penalty,
        where penalty is applied if time > max_time.
        """
        t = self.calculate_student_time(route)
        p = sum(self.orders[oid]['profit'] for oid in route)
        component = p - t * self.cost_per_second
        if t > self.max_time:
            component -= (t - self.max_time) * 100
        return component, t, p

    def calculate_student_time(self, route):
        if not route:
            return 0
        t = 0
        # Use list comprehension to extract nodes.
        nodes = [self.orders[order_id]['node'] for order_id in route]
        t += self.driving_times[self.headquarters][nodes[0]]
        for i in range(len(nodes) - 1):
            t += self.driving_times[nodes[i]][nodes[i + 1]]
        t += self.driving_times[nodes[-1]][self.headquarters]
        t += sum(self.orders[order_id]['duration'] * 60 for order_id in route)
        return t

    def compute_overall_fitness(self):
        total = 0
        for i in range(self.student_count):
            comp, t, p = self.evaluate_route(self.cur_solution[i])
            self.route_fitness[i] = comp
            self.route_time[i] = t
            self.route_profit[i] = p
            total += comp
        return total

    def initial_solution_random(self):
        solution = [[] for _ in range(self.student_count)]
        orders_list = list(self.orders.values())
        random.shuffle(orders_list)
        for order in orders_list:
            eligible = order['eligible']
            if not eligible:
                continue
            student = random_choice_set(eligible)
            pos = random.randint(0, len(solution[student]))
            candidate_route = solution[student].copy()
            candidate_route.insert(pos, order['id'])
            if self.calculate_student_time(candidate_route) <= self.max_time:
                solution[student] = candidate_route
        return solution

    def generate_neighbor(self, solution):
        op_probs = {
            'add': 0.25,
            'remove': 0.15,
            '2-opt': 0.25,
            'swap': 0.15,
            'relocate': 0.20
        }
        operators = list(op_probs.keys())
        weights = list(op_probs.values())
        operation = random.choices(operators, weights=weights, k=1)[0]
        self.op_attempts[operation] += 1

        if operation == 'add':
            assigned_ids = set()
            for route in solution:
                assigned_ids.update(route)
            unassigned = [oid for oid in self.orders if oid not in assigned_ids]
            if not unassigned:
                return None, None
            order_id = random.choice(unassigned)
            order = self.orders[order_id]
            eligible = order['eligible']
            if not eligible:
                return None, None
            student = random_choice_set(eligible)
            pos = random.randint(0, len(solution[student]))
            new_route = solution[student][:]  # candidate route for that student
            new_route.insert(pos, order_id)
            if self.calculate_student_time(new_route) <= self.max_time:
                # Return extra information (pos and order_id) for incremental update.
                return ('add', student, new_route, pos, order_id), operation
            return None, None

        elif operation == 'remove':
            student = random.randint(0, self.student_count - 1)
            if not solution[student]:
                return None, None
            new_route = solution[student][:]
            idx = random.randint(0, len(new_route) - 1)
            removed_order = new_route[idx]
            new_route.pop(idx)
            if self.calculate_student_time(new_route) <= self.max_time:
                # Return extra info (idx and removed order) for incremental update.
                return ('remove', student, new_route, idx, removed_order), operation
            return None, None

        elif operation == '2-opt':
            student = random.randint(0, self.student_count - 1)
            route = solution[student]
            if len(route) < 3:
                return None, None
            i = random.randint(0, len(route) - 3)
            j = random.randint(i + 2, len(route) - 1)
            new_route = route[:i+1] + list(reversed(route[i+1:j+1])) + route[j+1:]
            if self.calculate_student_time(new_route) <= self.max_time:
                # Return indices i and j for incremental update.
                return ('2-opt', student, new_route, i, j), operation
            return None, None

        elif operation == 'swap':
            # 50% chance: swap orders between two different routes.
            if random.random() < 0.50 and len([i for i in range(self.student_count) if solution[i]]) >= 2:
                valid_students = [i for i in range(self.student_count) if solution[i]]
                s1, s2 = random.sample(valid_students, 2)
                if not solution[s1] or not solution[s2]:
                    return None, None
                i1 = random.randint(0, len(solution[s1]) - 1)
                i2 = random.randint(0, len(solution[s2]) - 1)
                r1 = solution[s1].copy()
                r2 = solution[s2].copy()
                o1, o2 = r1[i1], r2[i2]
                # Check eligibility.
                if s1 not in self.orders[o2]['eligible'] or s2 not in self.orders[o1]['eligible']:
                    return None, None
                r1[i1], r2[i2] = r2[i2], r1[i1]
                if (self.calculate_student_time(r1) <= self.max_time and
                    self.calculate_student_time(r2) <= self.max_time):
                    return ('swap_between', s1, s2, r1, r2), operation
                return None, None
            else:
                # Swap within a single route.
                valid_students = [i for i in range(self.student_count) if len(solution[i]) >= 2]
                if not valid_students:
                    return None, None
                s = random.choice(valid_students)
                i1, i2 = random.sample(range(len(solution[s])), 2)
                new_route = solution[s].copy()
                new_route[i1], new_route[i2] = new_route[i2], new_route[i1]
                if self.calculate_student_time(new_route) <= self.max_time:
                    return ('swap_within', s, new_route, i1, i2), operation
                return None, None

        elif operation == 'relocate':
            valid_s1 = [i for i in range(self.student_count) if solution[i]]
            if not valid_s1:
                return None, None
            s1 = random.choice(valid_s1)
            i1 = random.randint(0, len(solution[s1]) - 1)
            order_id = solution[s1][i1]
            eligible = self.orders[order_id]['eligible'] - {s1}
            if not eligible:
                return None, None
            s2 = random_choice_set(eligible)
            i2 = random.randint(0, len(solution[s2]))
            r1 = solution[s1].copy()
            r2 = solution[s2].copy()
            r1.pop(i1)
            r2.insert(i2, order_id)
            if (self.calculate_student_time(r1) <= self.max_time and
                self.calculate_student_time(r2) <= self.max_time):
                return ('relocate', s1, s2, r1, r2, i1, i2, order_id), operation
            return None, None

        return None, None

    def incremental_update_insertion(self, student, pos, order_id):
        order = self.orders[order_id]
        node_new = order['node']
        service_time = order['duration'] * 60
        current_route = self.cur_solution[student]
        dt_matrix = self.driving_times
        hq = self.headquarters
        # Compute delta time for insertion based on position.
        if not current_route:
            delta_time = (dt_matrix[hq][node_new] +
                          dt_matrix[node_new][hq] + service_time)
        else:
            if pos == 0:
                first_order = current_route[0]
                node_first = self.orders[first_order]['node']
                delta_time = (dt_matrix[hq][node_new] +
                              dt_matrix[node_new][node_first] -
                              dt_matrix[hq][node_first] + service_time)
            elif pos == len(current_route):
                last_order = current_route[-1]
                node_last = self.orders[last_order]['node']
                delta_time = (dt_matrix[node_last][node_new] +
                              dt_matrix[node_new][hq] -
                              dt_matrix[node_last][hq] + service_time)
            else:
                prev_order = current_route[pos - 1]
                next_order = current_route[pos]
                node_prev = self.orders[prev_order]['node']
                node_next = self.orders[next_order]['node']
                delta_time = (dt_matrix[node_prev][node_new] +
                              dt_matrix[node_new][node_next] -
                              dt_matrix[node_prev][node_next] + service_time)
        delta_profit = order['profit']
        # Use cached values.
        old_fitness = self.route_fitness[student]
        old_time = self.route_time[student]
        old_profit = self.route_profit[student]
        new_time = old_time + delta_time
        new_profit = old_profit + delta_profit
        old_penalty = 100 * max(0, old_time - self.max_time)
        new_penalty = 100 * max(0, new_time - self.max_time)
        new_fitness = new_profit - new_time * self.cost_per_second - new_penalty
        return new_fitness

    def incremental_update_removal(self, student, pos, order_id):
        order = self.orders[order_id]
        node_removed = order['node']
        service_time = order['duration'] * 60
        current_route = self.cur_solution[student]
        dt_matrix = self.driving_times
        hq = self.headquarters
        if len(current_route) == 1:
            delta_time = -(dt_matrix[hq][node_removed] +
                           dt_matrix[node_removed][hq] + service_time)
        else:
            if pos == 0:
                next_order = current_route[1]
                node_next = self.orders[next_order]['node']
                old_segment = (dt_matrix[hq][node_removed] +
                               dt_matrix[node_removed][node_next] + service_time)
                new_segment = dt_matrix[hq][node_next]
                delta_time = new_segment - old_segment
            elif pos == len(current_route) - 1:
                prev_order = current_route[-2]
                node_prev = self.orders[prev_order]['node']
                old_segment = (dt_matrix[node_prev][node_removed] +
                               dt_matrix[node_removed][hq] + service_time)
                new_segment = dt_matrix[node_prev][hq]
                delta_time = new_segment - old_segment
            else:
                prev_order = current_route[pos - 1]
                next_order = current_route[pos + 1]
                node_prev = self.orders[prev_order]['node']
                node_next = self.orders[next_order]['node']
                old_segment = (dt_matrix[node_prev][node_removed] +
                               dt_matrix[node_removed][node_next] + service_time)
                new_segment = dt_matrix[node_prev][node_next]
                delta_time = new_segment - old_segment
        delta_profit = -order['profit']
        old_time = self.route_time[student]
        old_profit = self.route_profit[student]
        new_time = old_time + delta_time
        new_profit = old_profit + delta_profit
        old_penalty = 100 * max(0, old_time - self.max_time)
        new_penalty = 100 * max(0, new_time - self.max_time)
        new_fitness = new_profit - new_time * self.cost_per_second - new_penalty
        return new_fitness

    def incremental_update_2opt(self, student, route, i, j):
        """
        Incrementally update the fitness for a 2-opt move.
        The new route is: route[:i+1] + reversed(route[i+1:j+1]) + route[j+1:].
        This function uses the cached route time and profit.
        """
        dt_matrix = self.driving_times
        orders = self.orders
        hq = self.headquarters
        cost_sec = self.cost_per_second

        old_time = self.route_time[student]
        old_profit = self.route_profit[student]
        # Build list of nodes for the current route.
        nodes = [orders[oid]['node'] for oid in route]

        # Identify affected segments.
        # First affected segment: from route[i] -> route[i+1]
        old_cost_first = dt_matrix[nodes[i]][nodes[i+1]]
        # Last affected segment: from route[j] -> (headquarters or next node)
        if j == len(route) - 1:
            old_cost_last = dt_matrix[nodes[j]][hq]
        else:
            old_cost_last = dt_matrix[nodes[j]][nodes[j+1]]
        # Internal segments cost (unchanged if not reversed, but will change order):
        old_internal = 0
        for k in range(i+1, j):
            old_internal += dt_matrix[nodes[k]][nodes[k+1]]
        old_segment_total = old_cost_first + old_internal + old_cost_last

        # New segments after reversal:
        new_cost_first = dt_matrix[nodes[i]][nodes[j]]
        if j == len(route) - 1:
            new_cost_last = dt_matrix[nodes[i+1]][hq]
        else:
            new_cost_last = dt_matrix[nodes[i+1]][nodes[j+1]]
        # Compute internal cost for reversed segment.
        reversed_segment = list(reversed(nodes[i+1:j+1]))
        new_internal = 0
        for k in range(len(reversed_segment) - 1):
            new_internal += dt_matrix[reversed_segment[k]][reversed_segment[k+1]]
        new_segment_total = new_cost_first + new_internal + new_cost_last

        delta = new_segment_total - old_segment_total
        new_time = old_time + delta

        # Recalculate penalty differences.
        old_penalty = 100 * max(0, old_time - self.max_time)
        new_penalty = 100 * max(0, new_time - self.max_time)
        new_fitness = old_profit - new_time * cost_sec - new_penalty

        return new_fitness, new_time, old_profit

    def p_accept(self, candidate_fitness):
        try:
            return math.exp((candidate_fitness - self.cur_fitness) / self.T)
        except OverflowError:
            return 1

    def anneal(self):
        # Initialize solution and cached per-route fitness.
        self.cur_solution = self.initial_solution_random()
        self.cur_fitness = self.compute_overall_fitness()
        self.best_solution = [r.copy() for r in self.cur_solution]
        self.best_fitness = self.cur_fitness

        accepted, rejected = 0, 0
        # Counters for the current debug window
        window_accepted, window_rejected = 0, 0
        debug_interval = 2
        last_debug_time = time.time()
        start_time = last_debug_time

        # Define a reheat factor to slightly increase temperature when needed.
        reheat_factor = 1.07

        # Cache frequently used attributes to reduce inner-loop overhead.
        orders = self.orders
        dt_matrix = self.driving_times
        cost_sec = self.cost_per_second
        hq = self.headquarters

        while self.T >= self.stopping_T and self.iteration < self.stopping_iter:
            result = self.generate_neighbor(self.cur_solution)
            if result is None or result[0] is None:
                self.iteration += 1
                continue

            neighbor_info, op_used = result
            op_type = neighbor_info[0]

            if op_type == '2-opt':
                student_idx = neighbor_info[1]
                new_route = neighbor_info[2]
                i, j = neighbor_info[3], neighbor_info[4]
                old_comp = self.route_fitness[student_idx]
                new_comp, new_time, profit = self.incremental_update_2opt(student_idx, self.cur_solution[student_idx], i, j)
                candidate_fitness = self.cur_fitness - old_comp + new_comp
                if candidate_fitness > self.cur_fitness or random.random() < self.p_accept(candidate_fitness):
                    self.cur_solution[student_idx] = new_route
                    self.route_fitness[student_idx] = new_comp
                    self.route_time[student_idx] = new_time
                    self.route_profit[student_idx] = profit
                    self.cur_fitness = candidate_fitness
                    accepted += 1
                    window_accepted += 1
                    if candidate_fitness > self.best_fitness:
                        self.best_solution = [r.copy() for r in self.cur_solution]
                        self.best_fitness = candidate_fitness
                else:
                    rejected += 1
                    window_rejected += 1

            # For operators that modify two routes.
            elif op_type in ['swap_between', 'relocate']:
                if op_type == 'swap_between':
                    s1, s2 = neighbor_info[1], neighbor_info[2]
                    new_route1, new_route2 = neighbor_info[3], neighbor_info[4]
                    old_comp1 = self.route_fitness[s1]
                    old_comp2 = self.route_fitness[s2]
                    # For now, we recalc full route evaluations.
                    new_comp1, t1, p1 = self.evaluate_route(new_route1)
                    new_comp2, t2, p2 = self.evaluate_route(new_route2)
                    candidate_fitness = self.cur_fitness - old_comp1 - old_comp2 + new_comp1 + new_comp2
                    if candidate_fitness > self.cur_fitness or random.random() < self.p_accept(candidate_fitness):
                        self.cur_solution[s1] = new_route1
                        self.cur_solution[s2] = new_route2
                        self.route_fitness[s1] = new_comp1
                        self.route_fitness[s2] = new_comp2
                        self.route_time[s1], self.route_time[s2] = t1, t2
                        self.route_profit[s1], self.route_profit[s2] = p1, p2
                        self.cur_fitness = candidate_fitness
                        accepted += 1
                        window_accepted += 1
                        if candidate_fitness > self.best_fitness:
                            self.best_solution = [r.copy() for r in self.cur_solution]
                            self.best_fitness = candidate_fitness
                    else:
                        rejected += 1
                        window_rejected += 1
                else:  # relocate
                    s1, s2 = neighbor_info[1], neighbor_info[2]
                    new_route1, new_route2 = neighbor_info[3], neighbor_info[4]
                    # For relocate, we use full evaluation.
                    old_comp1 = self.route_fitness[s1]
                    old_comp2 = self.route_fitness[s2]
                    new_comp1, t1, p1 = self.evaluate_route(new_route1)
                    new_comp2, t2, p2 = self.evaluate_route(new_route2)
                    candidate_fitness = self.cur_fitness - old_comp1 - old_comp2 + new_comp1 + new_comp2
                    if candidate_fitness > self.cur_fitness or random.random() < self.p_accept(candidate_fitness):
                        self.cur_solution[s1] = new_route1
                        self.cur_solution[s2] = new_route2
                        self.route_fitness[s1] = new_comp1
                        self.route_fitness[s2] = new_comp2
                        self.route_time[s1], self.route_time[s2] = t1, t2
                        self.route_profit[s1], self.route_profit[s2] = p1, p2
                        self.cur_fitness = candidate_fitness
                        accepted += 1
                        window_accepted += 1
                        if candidate_fitness > self.best_fitness:
                            self.best_solution = [r.copy() for r in self.cur_solution]
                            self.best_fitness = candidate_fitness
                    else:
                        rejected += 1
                        window_rejected += 1

            # For operators that modify a single route (add, remove, swap_within).
            else:
                student_idx = neighbor_info[1]
                new_route = neighbor_info[2]
                old_comp = self.route_fitness[student_idx]
                new_comp, t_new, p_new = self.evaluate_route(new_route)
                candidate_fitness = self.cur_fitness - old_comp + new_comp
                if candidate_fitness > self.cur_fitness or random.random() < self.p_accept(candidate_fitness):
                    self.cur_solution[student_idx] = new_route
                    self.route_fitness[student_idx] = new_comp
                    self.route_time[student_idx] = t_new
                    self.route_profit[student_idx] = p_new
                    self.cur_fitness = candidate_fitness
                    accepted += 1
                    window_accepted += 1
                    if candidate_fitness > self.best_fitness:
                        self.best_solution = [r.copy() for r in self.cur_solution]
                        self.best_fitness = candidate_fitness
                else:
                    rejected += 1
                    window_rejected += 1

            self.T *= self.alpha
            self.iteration += 1
            self.fitness_history.append(self.cur_fitness)
            self.temperature_history.append(self.T)
            self.op_success[op_used] += 1

            current_time = time.time()
            if current_time - last_debug_time >= debug_interval:
                iterations_per_sec = self.iteration / (current_time - start_time)
                total_ops = window_accepted + window_rejected
                window_acceptance_ratio = window_accepted / total_ops if total_ops > 0 else 0

                if total_ops > 0 and window_acceptance_ratio < 0.10:
                    old_T = self.T
                    self.T *= reheat_factor
                    print(f"Reheating: Acceptance ratio {window_acceptance_ratio:.2f} below threshold, increasing T from {old_T:.4f} to {self.T:.4f}")

                customers_serviced = sum(len(route) for route in self.cur_solution)
                print(f"Iteration: {self.iteration}, Temp: {self.T:.4f}, Profit: {self.cur_fitness:.2f}, "
                      f"Customers: {customers_serviced}, Best Profit: {self.best_fitness:.2f}, "
                      f"Iter/sec: {iterations_per_sec:.2f}, Acceptance ratio (window): {window_acceptance_ratio:.2f}")
                last_debug_time = current_time
                window_accepted, window_rejected = 0, 0

    def output_solution(self, filename='output.txt'):
        total_profit = sum(self.orders[oid]['profit'] for route in self.best_solution for oid in route)
        total_time = sum(self.calculate_student_time(route) for route in self.best_solution)
        total_cost = total_time * self.cost_per_second
        net_profit = total_profit - total_cost

        print("Output calculation:")
        print(f"- Total profit: {total_profit}")
        print(f"- Total cost: {total_cost}")
        print(f"- Net profit: {net_profit}")

        with open(filename, 'w') as f:
            f.write(f"{int(net_profit)}\n")
            for route in self.best_solution:
                f.write(f"{len(route)}\n")
                for oid in route:
                    f.write(f"{oid}\n")

    def visualize_progress(self):
        if not self.fitness_history or not hasattr(self, 'temperature_history') or len(self.temperature_history) != len(self.fitness_history):
            print("Temperature history not available for color grading. Falling back to simple plot.")
            x = list(range(len(self.fitness_history)))
            y = self.fitness_history
            plt.figure()
            plt.xlabel('Iteration')
            plt.ylabel('Net Profit (Fitness)')
            plt.title('Simulated Annealing Progress')
            plt.plot(x, y, color='purple', lw=2)
            plt.show()
        else:
            x = np.array(range(len(self.fitness_history)))
            y = np.array(self.fitness_history)
            t = np.array(self.temperature_history)
            log_t = np.log(t)
            norm = plt.Normalize(log_t.min(), log_t.max())
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap='viridis', norm=norm)
            lc.set_array(log_t[:-1])
            lc.set_linewidth(2)
            fig, ax = plt.subplots()
            ax.add_collection(lc)
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(y.min(), y.max() * 1.1)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Net Profit (Fitness)')
            ax.set_title('Simulated Annealing Progress with Temperature Grading')
            fig.colorbar(lc, ax=ax, label='Log Temperature')
            plt.show()

        op_usage = {op: self.op_attempts[op] / max(1, self.iteration) for op in self.operators}
        op_success = {op: self.op_success[op] / max(1, self.op_attempts[op]) for op in self.operators}
        ops = list(self.operators)
        usage_vals = [op_usage[op] for op in ops]
        success_vals = [op_success[op] for op in ops]
        x_axis = np.arange(len(ops))
        width = 0.35

        plt.figure()
        plt.bar(x_axis - width/2, usage_vals, width, label='Usage Frequency')
        plt.bar(x_axis + width/2, success_vals, width, label='Success Rate')
        plt.xlabel('Operator')
        plt.ylabel('Rate')
        plt.title('Operator Statistics')
        plt.xticks(x_axis, ops, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()

        orders_per_student = [len(route) for route in self.best_solution]
        students = list(range(self.student_count))

        plt.figure()
        plt.bar(students, orders_per_student, alpha=0.7, label='Orders per Student')
        plt.xlabel('Student')
        plt.ylabel('Number of Orders')
        plt.title('Orders per Student')
        plt.legend()
        plt.tight_layout()
        plt.show()

        route_profits = [sum(self.orders[oid]['profit'] for oid in route)
                         for route in self.best_solution]
        route_times = [self.calculate_student_time(route) / 60
                       for route in self.best_solution]
        route_net_profits = []
        for i, route in enumerate(self.best_solution):
            total_time = self.calculate_student_time(route)
            total_cost = total_time * self.cost_per_second
            net_profit = route_profits[i] - total_cost
            route_net_profits.append(net_profit)

        plt.figure(figsize=(10, 6))
        plt.plot(students, route_profits, 'ro-', label='Profit')
        plt.plot(students, route_times, 'bs-', label='Time (min)')
        plt.plot(students, route_net_profits, 'g^-', label='Net Profit')
        plt.xlabel('Student')
        plt.ylabel('Value')
        plt.title('Route Distribution (Profit, Time & Net Profit)')
        plt.legend()

        max_val = max(max(route_profits), max(route_times), max(route_net_profits))
        plt.ylim(0, max_val * 1.2)
        plt.tight_layout()
        plt.show()

def read_orders(filename):
    orders = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip header
        for row in reader:
            order_id = int(row[0])
            node_id = int(row[2])
            duration = int(row[5])
            profit = int(row[6])
            eligible = [int(val) for val in row[7:27]]
            eligible_students = {i for i, val in enumerate(eligible) if val == 1}
            orders.append({
                'id': order_id,
                'node': node_id,
                'duration': duration,
                'profit': profit,
                'eligible': eligible_students,
                'service_time': duration * 60
            })
    return orders

def read_driving_times(filename):
    driving_times = []
    with open(filename, 'r') as f:
        for line in f:
            times = list(map(int, line.strip().split()))
            driving_times.append(times)
    return driving_times

if __name__ == '__main__':
    orders = read_orders('orders.txt')
    driving_times = read_driving_times('drivingtimes.txt')

    sa = SimAnneal(orders, driving_times)
    sa.anneal()
    sa.output_solution()
    sa.visualize_progress()
