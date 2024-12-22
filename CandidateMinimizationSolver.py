from typing import List

def solve_sudoku_int(board: List[List[int]]) ->int:
        """
        Solve the Sudoku puzzle represented as a List[List[int]] in place.
        
        Args:
            board (List[List[int]]): A 9x9 grid where 0 represents empty cells.

        Returns:
            bool: True if the board is solved, False otherwise.
        """
        # Track empty cells and constraints for rows, columns, and boxes
        unsolved = []
        row = [set() for _ in range(9)]
        col = [set() for _ in range(9)]
        box = [set() for _ in range(9)]

        # Populate the constraints and track empty cells
        for i in range(9):
            for j in range(9):
                val = board[i][j]
                if val != 0:  # If the cell is filled
                    box_index = (i // 3) * 3 + (j // 3)
                    if (
                        val in row[i]
                        or val in col[j]
                        or val in box[box_index]
                    ):
                        return -1
                    else:
                        # Add value to constraints
                        row[i].add(val)
                        col[j].add(val)
                        box[box_index].add(val)
                else:
                    unsolved.append((i, j))


        # Backtracking function with minimum candidate optimization
        def backtrack(index: int) -> bool:
            if index == len(unsolved):
                return True  # Solved
            
            # Find the cell with the minimum candidates
            min_candidates = 10
            min_candidate_index = index
            for x in range(index, len(unsolved)):
                r, c = unsolved[x]
                box_index = (r // 3) * 3 + (c // 3)
                possible = (
                    {1, 2, 3, 4, 5, 6, 7, 8, 9}
                    - row[r]
                    - col[c]
                    - box[box_index]
                )
                candidates = len(possible)
                if candidates < min_candidates:
                    min_candidates = candidates
                    min_candidate_index = x
                if min_candidates == 1:  # Early exit if only one candidate
                    break

            # Swap the cell with the fewest candidates to the current position
            unsolved[index], unsolved[min_candidate_index] = (
                unsolved[min_candidate_index],
                unsolved[index],
            )

            r, c = unsolved[index]
            box_index = (r // 3) * 3 + (c // 3)
            possible_numbers = (
                {1, 2, 3, 4, 5, 6, 7, 8, 9}
                - row[r]
                - col[c]
                - box[box_index]
            )

            # Try each possible number
            for num in possible_numbers:
                board[r][c] = num
                row[r].add(num)
                col[c].add(num)
                box[box_index].add(num)

                if backtrack(index + 1):
                    return True

                # Undo the placement
                board[r][c] = 0
                row[r].remove(num)
                col[c].remove(num)
                box[box_index].remove(num)

            return False

        if not backtrack(0):
            return -2  # No solution exists
        return 1  # Solved