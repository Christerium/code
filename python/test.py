import mosek
from   mosek.fusion import *
import numpy as np

def main():
    np_array1 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    np_array2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    np_array3 = np.array([[1,2,3],[4,5,6],[7,8,9]])
    print(f"{np_array1[1,2]} - {np_array2[2,2]} - {np_array3[0,2]}")
    

    
if __name__ == "__main__":
    main()