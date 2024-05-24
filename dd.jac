"""Snake, classic arcade game.\n\nExercises\n\n1. How do you make the snake faster or slower?\n2. How can you make the snake go around the edges?\n3. How would you move the food?\n4. Change the snake to respond to mouse clicks.\n"""

import:py os ;
import:py random ;
import:py from random, randrange ;
include:py turtle ;
import:py from freegames, square, vector ;
import:py from flappy6, add ;

with entry {
    print(add(89, 7)) ;    
    food = vector(0, 0);    
    snake = [vector(10, 0)];    
    aim = vector(0, -10);
}

can change(x: Any, y: Any) {
    """Change snake direction.""" ;    
    aim.x = x;    
    aim.y = y;
}

can inside(head: Any) {
    """Return True if head inside boundaries.""" ;    
    return -200 < head.x< 190 and -200 < head.y< 190 ;
}

can move() {
    """Move snake forward one segment.""" ;    
    head = snake[-1].copy();    
    head.move(aim) ;    
    if not inside(head) or head in snake {
        square(head.x, head.y, 9, "red") ;    
        update() ;    
        return ;}
    
    snake.append(head) ;    
    if head == food {
        print("Snake:", len(snake)) ;    
        food.x = (randrange(-15, 15) * 10);    
        food.y = (randrange(-15, 15) * 10);} else {
        snake.pop(0) ;
    }
    
    clear() ;    
    
    for body in snake {
        square(body.x, body.y, 9, "black") ;
    }
    
    square(food.x, food.y, 9, "green") ;    
    update() ;    
    ontimer(move, 100) ;
}

with entry {
    setup(420, 420, 370, 0) ;    
    hideturtle() ;    
    tracer(False) ;    
    listen() ;    
    onkey(with  can change(10, 0), "Right") ;    
    onkey(with  can change(-10, 0), "Left") ;    
    onkey(with  can change(0, 10), "Up") ;    
    onkey(with  can change(0, -10), "Down") ;    
    move() ;    
    done() ;
}

