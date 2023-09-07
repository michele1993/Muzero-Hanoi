def hanoi_solver(disks, goal_peg=2):
    """ Solve the tower of Hanoi starting from any (legal) configuration
        Args:
            disks: initial configuration
            goal_peg: the goal peg, where you want to build the tower, 2= rightmost peg
        Returns:    
            the min n. of moves necessary to solve the task from the given config.
    """
    n = len(disks)
    # Calculate the next target peg for each of the disks
    target = goal_peg
    targets = [0] * n # need this for loop below
    for i in range(n-1,-1,-1):
        targets[i] = target
        if disks[i] != target:
            # To allow for this move, the smaller disk needs to get out of the way
            target = 3 - target - disks[i]
        
    i=0
    move_counter=0
    while i <n: # Not yet solved ?
        # Find the disk that should move
        for i in range(n):
            if targets[i] != disks[i]:
                target = targets[i]
                ## ====== Uncomment if want to print moves =====
                #print(f"moved disk {i} from peg {disks[i]} to {target}")
                ## ===========================================
                move_counter +=1
                disks[i] = target # Make move
                # Update the next targets of the smaller disks
                for j in range(i-1,-1,-1):
                    targets[j] = target
                    target = 3 - target - disks[j]
                break
            i+=1 # if all targets match, add 1 to i to terminate while loop   
    return move_counter
