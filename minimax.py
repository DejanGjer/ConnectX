def my_agent_alpha_beta(obs, config):
    # Your code here: Amend the agent!
    
    import random
    import numpy as np
    
    #==========================================================================================
    
    # Gets board at next step if agent drops piece in selected column
    def drop_piece(grid, col, mark, config):
        next_grid = grid.copy()
        for row in range(config.rows-1, -1, -1):
            if next_grid[row][col] == 0:
                break
        next_grid[row][col] = mark
        return next_grid

    # Helper function for get_heuristic: checks if window satisfies heuristic conditions
    def check_window(window, piece, config):
        ans = np.array([0, 0, 0, 0])
        if window.count(piece) == config.inarow and window.count(0) == 0:
            ans[0] = 1
        elif window.count(piece) == config.inarow - 1 and window.count(0) == 1:
            ans[1] = 1
        else:
            piece = piece % 2 + 1
            if window.count(piece) == config.inarow and window.count(0) == 0:
                ans[2] = 1
            elif window.count(piece) == config.inarow - 1 and window.count(0) == 1:
                ans[3] = 1
        return ans
        
    

    def count_windows(grid, piece, config):
        #num_windows = 0
        ans = np.array([0,0,0,0])
        # horizontal
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[row, col:col+config.inarow])
                ans += check_window(window, piece, config)
    
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row+config.inarow, col])
                ans += check_window(window, piece, config)
    
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                ans += check_window(window, piece, config)
    
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                ans += check_window(window, piece, config)
        
        return ans
    
    #==========================================================================================
    
    #Heuristika
    def get_heuristic(grid, mark, config):
        nums = count_windows(grid, mark, config)
        #num_threes = count_windows(grid, 3, mark, config)
        #num_fours = count_windows(grid, 4, mark, config)
        #num_threes_opp = count_windows(grid, 3, mark%2+1, config)
        #num_fours_opp = count_windows(grid, 4, mark%2+1, config)
        score = nums[1] - 1e2*nums[3] - 1e4*nums[2] + 1e6*nums[0]
        return score
    
    #Poziva minimax
    def score_move(grid, col, mark, config, nsteps):
        next_grid = drop_piece(grid, col, mark, config)
        score = minimax(next_grid, nsteps-1, False, mark, config, -np.Inf, np.Inf)
        return score
    
    #==========================================================================================

    # Helper function for minimax: checks if agent or opponent has four in a row in the window
    def is_terminal_window(window, config):
        return window.count(1) == config.inarow or window.count(2) == config.inarow
    
    # Helper function for minimax: checks if game has ended
    def is_terminal_node(grid, config):
        # Check for draw 
        if list(grid[0, :]).count(0) == 0:
            return True
        # Check for win: horizontal, vertical, or diagonal
        # horizontal 
        for row in range(config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[row, col:col+config.inarow])
                if is_terminal_window(window, config):
                    return True
        # vertical
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns):
                window = list(grid[row:row+config.inarow, col])
                if is_terminal_window(window, config):
                    return True
        # positive diagonal
        for row in range(config.rows-(config.inarow-1)):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row+config.inarow), range(col, col+config.inarow)])
                if is_terminal_window(window, config):
                    return True
        # negative diagonal
        for row in range(config.inarow-1, config.rows):
            for col in range(config.columns-(config.inarow-1)):
                window = list(grid[range(row, row-config.inarow, -1), range(col, col+config.inarow)])
                if is_terminal_window(window, config):
                    return True
        return False
    
    #==========================================================================================
    
    # Minimax implementation
    def minimax(node, depth, maximizingPlayer, mark, config, alpha, beta):
        is_terminal = is_terminal_node(node, config)
        valid_moves = [c for c in range(config.columns) if node[0][c] == 0]
        if depth == 0 or is_terminal:
            return get_heuristic(node, mark, config)
        if maximizingPlayer:
            value = -np.Inf
            for col in valid_moves:
                child = drop_piece(node, col, mark, config)
                temp = minimax(child, depth-1, False, mark, config, alpha, beta)
                value = max(value, temp)
                alpha = max(alpha, temp)
                if beta <= alpha:
                    break
            return value
        else:
            value = np.Inf
            for col in valid_moves:
                child = drop_piece(node, col, mark%2+1, config)
                temp = minimax(child, depth-1, True, mark, config, alpha, beta)
                value = min(value, temp)
                beta = min(beta, temp)
                if beta <= alpha:
                    break
            return value
        
    #==========================================================================================

    N_STEPS = 3
    
    # Get list of valid moves
    valid_moves = [c for c in range(config.columns) if obs.board[c] == 0]
    # Convert the board to a 2D grid
    grid = np.asarray(obs.board).reshape(config.rows, config.columns)
    # Use the heuristic to assign a score to each possible board in the next step
    scores = dict(zip(valid_moves, [score_move(grid, col, obs.mark, config, N_STEPS) for col in valid_moves]))
    # Get a list of columns (moves) that maximize the heuristic
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select at random from the maximizing columns
    return random.choice(max_cols)
