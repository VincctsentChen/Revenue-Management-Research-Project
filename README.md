# Revenue-Management-Research-Project
The Julia and Python codes for optimal revenue management
Corporates always want to find the optimal pricing strategy that maximizes their profits. However, since the actual function between price and sales is unknown,
we need to use advanced algorithm to find such relatinship. In this project, we divide the whole price space into small invertals, and use Taylor expansion to simulate the relationship within each interval.
We leverage MIP(mixed-integer-programming) to construct the optimization model to obtain the optimal pricing. In addition, we prove that the pricing produced by our algorithm is consistent to the actually optimal pricing. 
