import random 
import timeit

INTERVAL= 1000

random.seed(42)  

def compute_points():
    
    random.seed(42)  
    
    circle_points= 0

    # Total Random numbers generated= possible x 
    # values* possible y values 
    for i in range(INTERVAL**2): 
      
        # Randomly generated x and y values from a 
        # uniform distribution 
        # Rannge of x and y values is -1 to 1 
        
        
        rand_x= random.uniform(-1, 1) 
        rand_y= random.uniform(-1, 1) 
      
        # Distance between (x, y) from the origin 
        origin_dist= rand_x**2 + rand_y**2
      
        # Checking if (x, y) lies inside the circle 
        if origin_dist<= 1: 
            circle_points+= 1
      
        # Estimating value of pi, 
        # pi= 4*(no. of points generated inside the  
        # circle)/ (no. of points generated inside the square) 
    
     
    
    return circle_points

start = timeit.default_timer()
circle_points = compute_points()
end = timeit.default_timer()


pi = 4* circle_points/ INTERVAL**2 
print("Circle points number :",circle_points)
print("Final Estimation of Pi=", pi, "cpu time :",end-start) 