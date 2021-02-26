def monte_carlo_agent(obs, config):
    import random
    import numpy as np
    import math
    import time

    global current_state
    init_time = time.time()
    t_max = config.timeout - 0.35
    cp_default = 1

    # Gets board at next step if agent drops piece in selected column
    def drop_piece(grid, col, mark, config):
        next_grid = grid.copy()
        for row in range(config.rows-1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = mark
        return next_grid

    def is_tie(grid):
        if list(grid[0, :]).count(0) == 0:
            return True
        return False

    def is_terminal_window(window, config):
        return window.count(1) == config.inarow or window.count(2) == config.inarow

    # Helper function for minimax: checks if game has ended
    def is_win(grid, column, mark, config):
        columns = config.columns
        rows = config.rows
        inarow = config.inarow - 1
        row = min([r for r in range(rows) if grid[r, column] == mark])

        def count(offset_row, offset_column):
            for i in range(1, inarow + 1):
                r = row + offset_row * i
                c = column + offset_column * i
                if (
                        r < 0
                        or r >= rows
                        or c < 0
                        or c >= columns
                        or grid[r, c] != mark
                ):
                    return i - 1
            return inarow

        return (
                count(1, 0) >= inarow  # vertical.
                or (count(0, 1) + count(0, -1)) >= inarow  # horizontal.
                or (count(-1, -1) + count(1, 1)) >= inarow  # top left diagonal.
                or (count(-1, 1) + count(1, -1)) >= inarow  # top right diagonal.
        )

    def check_finish_and_score(grid, column, mark, config):
        if is_win(grid, column, mark, config):
            return (True, 1)
        elif is_tie(grid):
            return (True, 0.5)
        else:
            return (False, None)

    def uct_score(node_total_score, node_total_visits, parent_total_visits, cp=cp_default):
        """ UCB1 calculation. """
        if node_total_visits == 0:
            return math.inf
        return node_total_score / node_total_visits + cp * math.sqrt(
            2 * math.log(parent_total_visits) / node_total_visits)

    def opponent_mark(mark):
        """ The mark indicates which player is active - player 1 or player 2. """
        return 3 - mark

    def opponent_score(score):
        """ To backpropagate scores on the tree. """
        return 1 - score

    def random_action(grid, config):
        """ Returns a random legal action (from the open columns). """
        return random.choice([c for c in range(config.columns) if grid[0, c] == 0])

    def default_policy_simulation(grid, mark, config):
        """
        Run a random play simulation. Starting state is assumed to be a non-terminal state.
        Returns score of the game for the player with the given mark.
        """
        original_mark = mark
        column = random_action(grid, config)
        current_grid = drop_piece(grid, column, mark, config)
        is_terminal, score = check_finish_and_score(current_grid, column, mark, config)
        while not is_terminal:
            mark = opponent_mark(mark)
            column = random_action(current_grid, config)
            current_grid = drop_piece(current_grid, column, mark, config)
            is_terminal, score = check_finish_and_score(current_grid, column, mark, config)
        if mark == original_mark:
            return score
        else:
            return opponent_score(score)

    def find_action_taken_by_opponent(old_grid, new_grid, config):
        """ Given a new board state and a previous one, finds which move was taken. Used for recycling tree between moves. """
        for r in range(config.rows):
            for c in range(config.columns):
                if new_grid[r, c] != old_grid[r, c]:
                    return c
        return -1


    class State():

        def __init__(self, grid, mark, config, parent=None, is_terminal=False, terminal_score=None, action_taken=None):
            self.grid = grid.copy()
            self.mark = mark
            self.config = config
            self.parent = parent
            self.children = []
            self.node_total_score = 0
            self.node_total_visits = 0
            self.available_moves = [c for c in range(config.columns) if grid[0, c] == 0]
            self.expandable_moves = self.available_moves.copy()
            self.is_terminal = is_terminal
            self.terminal_score = terminal_score
            self.action_taken = action_taken

        def is_expandable(self):
            """ Checks if the node has unexplored children. """
            return (not self.is_terminal) and (len(self.expandable_moves) > 0)

        def choose_strongest_child(self, cp):
            """
            Chooses child that maximizes UCB1 score (Selection stage in the MCTS algorithm description).
            """
            uct_scores = np.array([uct_score(child.node_total_score, child.node_total_visits, self.node_total_visits, cp) for child in self.children])
            return self.children[uct_scores.argmax()]

        def expand_and_simulate_child(self):
            """
            Chooses child that maximizes UCB1 score (Selection stage in the MCTS algorithm description).
            """
            column = random.choice(self.expandable_moves)
            child_grid = drop_piece(self.grid, column, self.mark, self.config)
            is_terminal, terminal_score = check_finish_and_score(child_grid, column, self.mark, self.config)
            self.children.append(State(child_grid, opponent_mark(self.mark), self.config, parent=self, is_terminal=is_terminal, terminal_score=terminal_score, action_taken=column))

            simulation_score = self.children[-1].simulate()
            self.children[-1].backpropagate(simulation_score)
            self.expandable_moves.remove(column)

        def choose_play_child(self):
            """ Choose child with maximum total score."""
            children_scores = np.array([child.node_total_score for child in self.children])
            return self.children[children_scores.argmax()]

        def simulate(self):
            """
            Runs a simulation from the current state. 
            This method is used to simulate a game after move of current player, so if a terminal state was reached,
            the score would belong to the current player who made the move.
            But otherwise the score received from the simulation run is the opponent's score and thus needs to be flipped with the function opponent_score().            
            """
            if self.is_terminal:
                return self.terminal_score
            return opponent_score(default_policy_simulation(self.grid, self.mark, self.config))

        def backpropagate(self, simulation_score):
            """
            Backpropagates score and visit count to parents.
            """
            self.node_total_score += simulation_score
            self.node_total_visits += 1
            #current = self.parent
            #score = opponent_score(simulation_score)
            #while current is not None:
            #    current.node_total_score += score
            #    current.node_total_visits += 1
            #    current = current.parent
            #    score = opponent_score(score)
            if self.parent is not None:
                self.parent.backpropagate(opponent_score(simulation_score))

        def tree_single_run(self):
            """
            A single iteration of the 4 stages of the MCTS algorithm.
            """
            if self.is_terminal:
                self.backpropagate(self.terminal_score)
                return
            elif self.is_expandable():
                self.expand_and_simulate_child()
                return
            else:
                self.choose_strongest_child(cp_default).tree_single_run()

        def choose_child_via_action(self, action):
            """ Choose child given the action taken from the state. Used for recycling of tree. """
            for child in self.children:
                if child.action_taken == action:
                    return child
            return None

    grid = np.array(obs.board).reshape(config.rows, config.columns)
    mark = obs.mark

    # If current_state already exists, recycle it based on action taken by opponent
    try:
        current_state = current_state.choose_child_via_action(find_action_taken_by_opponent(grid, current_state.grid, config))
        current_state.parent = None
    # new game or other error in recycling attempt due to Kaggle mechanism
    except:
        current_state = State(grid, mark, config)

    # Run MCTS iterations until time limit is reached.
    while time.time() - init_time <= t_max:
        current_state.tree_single_run()

    current_state = current_state.choose_play_child()
    return current_state.action_taken
