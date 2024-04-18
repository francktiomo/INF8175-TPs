from player_abalone import PlayerAbalone
from game_state_abalone import GameStateAbalone
from seahorse.game.game_state import GameState
from seahorse.game.action import Action
from typing import Dict, List
from math import inf


def manhattanDist(A, B):
    """
    Calculates the Manhattan distance between two points in a 2D grid 
    as in compute_winner function of master_abalone.py
    
    Parameters:
        A (tuple): The coordinates of the first point in the format (x, y).
        B (tuple): The coordinates of the second point in the format (x, y).

    Returns:
        int: The Manhattan distance between the two points.
    """
    
    mask1 = [(0, 2), (1, 3), (2, 4)]
    mask2 = [(0, 4)]
    diff = (abs(B[0] - A[0]), abs(B[1] - A[1]))
    dist = (abs(B[0] - A[0]) + abs(B[1] - A[1])) / 2
    if diff in mask1:
        dist += 1
    if diff in mask2:
        dist += 2
    return dist


def normalize_manhattanDist(state: GameState) -> Dict[int, float]:
    """
    Computes the normalized Manhattan distance to center for each player.
    (Inspired by compute_winner function of master_abalone.py)

    Args:
        state (GameState): Current state of the game

    Returns:
        Dict[int, float]: Dictionary with player_id as key and normalized Manhattan distance to center as value
    """
    
    representation = state.get_rep()
    dimensions = representation.get_dimensions()
    center = (dimensions[0] // 2, dimensions[1] // 2)
    env = representation.get_env()
    players_ids = [player.get_id() for player in state.get_players()]
    distances = dict.fromkeys(players_ids, 0)
    pieces = dict.fromkeys(players_ids, 0)
    
    for i, j in list(env.keys()):
        p = env.get((i, j), None)
        if p.get_owner_id():
            distances[p.get_owner_id()] += manhattanDist(center, (i, j)) / 4
            pieces[p.get_owner_id()] += 1
    for player_id in players_ids:
        distances[player_id] /= pieces[player_id]
    return distances


def get_other_player(state: GameState, player: PlayerAbalone) -> PlayerAbalone:
    """
    Returns the other player.
    
    Args:
        state (GameState): Current state of the game
        player (PlayerAbalone): The player

    Returns:
        PlayerAbalone: The other player
    """
    players = state.get_players()
    
    if players[0] == player:
        return players[1]
    
    return players[0]


def compute_compactness(state: GameState) -> Dict[int, float]:
    """
    Computes the compactness of pieces for each player respectively.

    Args:
        state (GameState): Current state of the game

    Returns:
        Dict[int, float]: Dictionary with player_id as key and compactness as value
    """
    # Helper function to get the compactness score
    def get_compactness(state: GameState, player: PlayerAbalone) -> float:
        """
        Returns the compactness score for a player.

        Args:
            state (GameState): Current state of the game
            player (PlayerAbalone): The player

        Returns:
            float: compactness score
        """
        # Initialize total score
        total_neighbour_count = 0

        # Get the player's piece type
        player_piece_type = player.get_piece_type()
    
        # Retrieve positions of all pieces for the player
        pieces_positions = state.get_rep().get_pieces_player(player)[1]

    
        # Calculate the total number of matching neighbors for all pieces
        for piece in pieces_positions:
            # Get neighbors of the current piece
            neighbors = state.get_neighbours(piece[0], piece[1])

            # Count neighbors that match the player's piece type
            for neighbor in neighbors.values():
                if neighbor[0] == player_piece_type:
                    total_neighbour_count += 1

        # Normalizing the total score by the number of pieces multiplied by the maximum possible neighbors per piece
        if pieces_positions:
            compactness_score = total_neighbour_count / (len(pieces_positions) * 6)
        else:
            compactness_score = 0.0  # Avoid division by zero if pieces_pos is empty
    
        return compactness_score
    
    other = state.get_next_player() 
    player = get_other_player(state, other)
    other_id = other.get_id() 
    player_id = player.get_id()
    
    compactness_scores = {other_id: get_compactness(state, other),
                          player_id: get_compactness(state, player)}
    return compactness_scores


def abalone_heuristic(state: GameState) -> float:
    """
    Heuristic that combines the score, the distance to center and compactness.

    Args:
        state (GameState): Current state of the game

    Returns:
        float: Heuristic value
    """
    def compute_estimate(compactness: Dict[int, float], dist: Dict[int, float], score: Dict[int, float],  ) -> float:
        """
        Computes an estimate based on the given compactness, distance, and score.

        Args:
            compactness (Dict[int, float]): A dictionary mapping player IDs to their compactness scores.
            dist (Dict[int, float]): A dictionary mapping player IDs to their distance scores.
            score (Dict[int, float]): A dictionary mapping player IDs to their score.

        Returns:
            float: The computed estimate based on the given compactness, distance, and score.
        """
        estimate = 9 * compactness - dist + 9 * score
        return estimate
    
    player = state.get_next_player()
    player_id = player.get_id()
    other_player = get_other_player(state, player)
    other_player_id = other_player.get_id()
    dist = normalize_manhattanDist(state)
    compactness = compute_compactness(state)
    scores = state.scores
    
    # We want to maximize the difference between the estimates of the two players
    player_estimate = compute_estimate(compactness[player_id], dist[player_id], scores[player_id])
    other_estimate = compute_estimate(compactness[other_player_id], dist[other_player_id], scores[other_player_id])
        
    return player_estimate - other_estimate

def total_manhattan_distance(state: GameState) -> Dict[int, float]:
    """
    Computes the total manhattan distance to the center for each player.
    (Inspired from compute_distances_to_center function from master_abalone.py)
    
    Args:
        state (GameState): Current state of the game

    Returns:
        Dict[int, float]: Dictionary with player_id as key and distance to center as value
    """
    players_id = [player.get_id() for player in state.get_players()]
    final_rep = state.get_rep()
    env = final_rep.get_env()
    dim = final_rep.get_dimensions()
    dist = dict.fromkeys(players_id, 0)
    center = (dim[0] // 2, dim[1] // 2)
    for i, j in list(env.keys()):
        p = env.get((i, j), None)
        if p.get_owner_id():
            dist[p.get_owner_id()] += manhattanDist(center, (i, j))
    return dist


def compute_winner(state: GameState) -> List[PlayerAbalone]:
    """
    Computes the winners of the game based on the scores.
    (Inspired from compute_winner function from master_abalone.py)
    
    Args:
        state (GameState): Current state of the game

    Returns:
        Iterable[PlayerAbalone]: List of the players who won the game
    """
    scores = state.scores
    max_val = max(scores.values())
    players_id = list(filter(lambda key: scores[key] == max_val, scores))
    itera = list(filter(lambda x: x.get_id() in players_id, state.get_players()))
    
    if len(itera) > 1:  # There is a tie
        # Equal score (we compute the Manhattan distance to the center for each player
        # and take the player with the smallest Manhattan distance as the winner
        dist = total_manhattan_distance(state)
        min_dist = min(dist.values())
        players_id = list(filter(lambda key: dist[key] == min_dist, dist))
        itera = list(filter(lambda x: x.get_id() in players_id, state.get_players()))
        
    if len(itera) == 1:  # There is only one winner
        return itera[0]
    
    if len(itera) > 1:  # There is still a tie
        return None


def get_final_state(state: GameState) -> int:
    """
    Returns the score of the state for the max_player.

    Args:
        state (GameState): Current state of the game

    Returns:
        int: The score of the state for the max_player
    """
    winner = compute_winner(state)
    if winner is None:
        return 0
    if winner.get_id() == state.get_next_player().get_id():
        return inf
    else:
        return -inf

def piece_pushed(state: GameState, previous_state: GameState) -> bool:
    """
    Checks if a push happened between the previous state and the current state.

    Args:
        state (GameState): Current state of the game
        previous_state (GameState): Previous state of the game

    Returns:
        bool: any piece was pushed or not
    """
    player = state.get_next_player()
    
    # Previous positions of the given player's pieces
    prev_position = previous_state.get_rep().get_pieces_player(player)[1]
    
    # Current positions of the given player's pieces
    new_position = state.get_rep().get_pieces_player(player)[1]
    
    return not (set(new_position) == set(prev_position))


def search_score(state: GameState, depth: int, transposition_dictionary: Dict) -> ((str, bool), float):
    """
    Search the score of the state in the transposition dictionary.

    Args:
        state (GameState): Current state of the game
        depth (int): Depth of the tree
        transposition_dictionary (Dict): Transposition dictionary

    Returns:
        (str, bool): (state representation, game is near its end)
        float: score
    """
    # Determine if the current step + depth is greater than the max step
    could_end = state.step + depth > state.max_step
    
    # Construct the key for the transposition dictionary lookup
    transposition_key = (str(state.rep), could_end)
    
    # Attempt to retrieve the score from the transposition dictionary
    result = transposition_dictionary.get(transposition_key)
    if result is not None and result['depth'] >= depth:
        # Adjust the score based on the current player
        current_player = state.get_next_player()
        if result['player'] == current_player:
            score = result['score']
        else:
            score = -result['score']
        return transposition_key, score
    
    # Return the key and None if no suidictionary score was found
    return transposition_key, None


def get_state_score(
    *,
    state: GameState,
    previous_state: GameState,
    depth: int,
    heuristic,
    dictionary: Dict,
    alpha=-inf,
    beta=inf,
    quiescence_test: bool,
    quiescence_depth=1
    ) -> float:
    """
    Returns the score of the state based on negamax with alpha-beta pruning
    (This function is called recursively) 

    Args:
        state (GameState): Current state of the game
        previous_state (GameState): Previous state of the game
        depth (int): Depth of the tree
        heuristic (function): Heuristic function
        dictionary (Dict): Transposition dictionary
        alpha (int): Value of the best choice currently found for max player on the path from a node to the root
        beta (int): Value of the best choice currently found for the min player on the path from a node to the root
        quiescence_test (bool): Use quiescence search or not
        quiescence_depth (int): Depth of the quiescence search
        float: score of the state
    """

    # Get the score from the transposition dictionary
    dictionary_key, score_result = search_score(state, depth, dictionary)
    if score_result is not None:
        score = score_result
    elif state.is_done(): # Final state
        score = get_final_state(state)
        depth = inf
    elif depth == 0: # Max depth
        # use quiescence if a piece has been pushed
        if quiescence_test and piece_pushed(state, previous_state):
            score = get_state_score(
                state=state,
                previous_state=previous_state,
                depth=quiescence_depth,
                heuristic=heuristic,
                dictionary=dictionary,
                alpha=-beta,
                beta=-alpha,
                quiescence_test=False,
                quiescence_depth=quiescence_depth)
        else:
            score = heuristic(state)
    else: # Non final state below max depth
        score = -inf
        children = state.get_possible_actions()
        for child in children:
            child_score = get_state_score(
                state=child.get_next_game_state(),
                previous_state=state,
                depth=depth - 1,
                heuristic=heuristic,
                dictionary=dictionary,
                alpha=-beta,
                beta=-alpha,
                quiescence_test=quiescence_test,
                quiescence_depth=quiescence_depth)
            score = max(score, -child_score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
    # Update transposition dictionary
    dictionary[dictionary_key] = {"score": score, "depth": depth, "player": state.get_next_player()}

    return score



class MyPlayer(PlayerAbalone):
    """
    Player class for Abalone game. The player will use the negamax algorithm 
    with alpha-beta pruning.
    

    Attributes:
        piece_type (str): piece type of the player
    """

    def __init__(
        self, piece_type: str, name: str = "bob", time_limit: float = 60 * 15, *args
    ) -> None:
        """
        Initialize the PlayerAbalone instance.

        Args:
            piece_type (str): Type of the player's game piece
            name (str, optional): Name of the player (default is "bob")
            time_limit (float, optional): the time limit in (s)
        """
        super().__init__(piece_type, name, time_limit, *args)
        self.heuristic = abalone_heuristic
        self.search_depth = 3
        self.dictionary = {}
        self.quiescence_on = True
        self.quiescence_depth = 1

    def to_json(self):
        """
        A function that is mainly used for compatibility with the seahorse package

        Returns:
            str: ""
        """
        return ""

    def compute_action(self, current_state: GameState, **kwargs) -> Action:
        """
        Return the bestaction that the player will perform according to the heuristic

        Args:
            current_state (GameState): Current game state representation
            **kwargs: Additional keyword arguments

        Returns:
            Action: selected feasible action
        """
        # Don't use quiescence search and decrease search depth if the time is lower than 90s 
        if self.get_remaining_time() < 90:
            self.quiescence_on = False
            self.search_depth = 2
            
        # Get the score of the current state and update the transposition dictionary
        get_state_score(
            state=current_state,
            previous_state=current_state,
            depth=self.search_depth,
            heuristic=self.heuristic,
            dictionary=self.dictionary,
            quiescence_depth = self.quiescence_depth,
            quiescence_test=self.quiescence_on)

        # Compute the best action
        best_action = max(
            current_state.get_possible_actions(),
            key=lambda k: 
                -(search_score(k.get_next_game_state(), -inf, self.dictionary)[1] or inf))

        return best_action
