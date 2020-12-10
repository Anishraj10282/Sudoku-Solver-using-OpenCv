def printSudoku(grid):
    for i in range(0,9):
        for j in range(0,9):
            print(grid[i][j],end=" ")
        print()


def isSafe(grid, row, col, num):
    for i in range(0,9):
        if grid[row][i]==num:
            return False
        if grid[i][col]==num:
            return False
    secondary_row = row - row%3
    secondary_col = col - col%3
    
    for i in range(0,3):
        for j in range(0, 3):
            if(grid[i+secondary_row][j+secondary_col]==num):
                return False

    return True



def SolveSudoku(grid, row, col):
    if(row==8 and col==9):
        return True

    if (col==9):
        row += 1
        col = 0

    if(grid[row][col]>0):
        return SolveSudoku(grid, row, col+1)

    for num in range(1,10):
        if(isSafe(grid, row, col, num)):
            grid[row][col]=num

            if(SolveSudoku(grid, row, col+1)):
                return True

        grid[row][col]=0
    return False



if __name__=='__main__':
    
    grid = [[3, 0, 6, 5, 0, 8, 4, 0, 0], 
             [5, 2, 0, 0, 0, 0, 0, 0, 0], 
             [0, 8, 7, 0, 0, 0, 0, 3, 1], 
             [0, 0, 3, 0, 1, 0, 0, 8, 0], 
             [9, 0, 0, 8, 6, 3, 0, 0, 5], 
             [0, 5, 0, 0, 9, 0, 6, 0, 0], 
             [1, 3, 0, 0, 0, 0, 2, 5, 0], 
             [0, 0, 0, 0, 0, 0, 0, 7, 4], 
             [0, 0, 5, 2, 0, 6, 3, 0, 0]]

    printSudoku(grid)

    print()
    print()
    
    if(SolveSudoku(grid, 0, 0)):
        printSudoku(grid)
    else:
        print("No sudoku Exist")

    
