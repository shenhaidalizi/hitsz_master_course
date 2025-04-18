# Lecture10

**16**

| A    | B    |                 |
| ---- | ---- | --------------- |
| 15   | 46   | 46 = 15 * 3 + 1 |
| 15   | 1    | 15 = 15 * 1 + 0 |
| 0    | 1    | 1               |

1 = 46 - 15 * 3

$15^{-1} = 46 - 3 = 43$

**21**

b' = v' = 7, k' = 4, r' = 4, λ’=2.

Suppose the starter block is B = {2, 4, 5, 6};

| -    | 2    | 4    | 5    | 6    |
| ---- | ---- | ---- | ---- | ---- |
| 2    | 0    | 5    | 4    | 3    |
| 4    | 2    | 0    | 6    | 5    |
| 5    | 3    | 1    | 0    | 6    |
| 6    | 4    | 2    | 1    | 0    |

Examining this table we see that each of the non-zero integers 1, 2, 3, 4, 5, 6 in $Z_7$ occurs exactly twice in the off-diagonal positions and hence exactly twice as a difference. Hence, B is a difference set mod 7.

Then the blocks developed from B as a starter block,we have: B+0={2,4,5,6}, B+1={3,5,6,0}, B+2={4,6,0,1}, B+3={5,0,1,2}, B+4={6,1,2,3}, B+5={0,2,3,4}, B+6={1,3,4,5}

**28**

| -    | 0    | 1    | 3    | 9    |
| ---- | ---- | ---- | ---- | ---- |
| 0    | 0    | 12   | 10   | 4    |
| 1    | 1    | 0    | 11   | 5    |
| 3    | 3    | 2    | 0    | 7    |
| 9    | 9    | 8    | 6    | 0    |

Examining this table we see that each of the non-zero integers 1, 2, 3, 4, 5, 6,7,8,9,10,11,12 in Z13 occurs exactly once in the off-diagonal positions and hence exactly once as a difference. Hence, B is a difference set mod 13.

For that is a SBIBD, so the b = v = 13, k = r = 4, λ = 1;

B+0={0,1,3,9};

B+1={1,2,4,10}; 

B+2 = {2,3,5,11}; 

B+3 = {3,4,6,12}; 

B+4 = {4,5,7,0}; 

B+5={5,6,8,1};

B+6={6,7,9,2}; 

B+7={7,8,10,3}; 

B+11={11,12,1,7}; 

B+12={12,0,2,8}

**32**

We find two Steiner triple systems for a three one and a seven one;

Let X={a0,a1,a2}, Y={b0,b1,b2,b3,b4,b5,b6} are the sets of varieties. Let B1={(a0,a1,a2)}, B2={(b0,b1,b3),(b1,b2,b4),(b2,b3,b5),(b3,b4,b6),(b4,b5,b0),(b5,b6,b1),(b6,b0,b2)};

Then:

1. r = s = t,  (0,5,3),(5,9,12),(9,3,15),(3,12,18),(12,15,0),(15,18,5),(18,0,9) (1,6,8),(6,10,13),(10,8,16),(8,13,19),(13,16,1),(16,19,6),(19,1,10). (2,7,4),(7,11,14),(11,4,17),(4,14,20),(14,17,2),(17,20,7),(20,2,11)
2. i = j = k,  (0,1,2),(5,6,7),(9,10,11),(3,8,4),(12,13,14),(15,15,17),(18,19,20)
3.  i,j,k is differnt from each other, r,s,t is different from each other,hence there are 42 triples. Including: (0,6,4),(0,7,8),(1,5,4),(1,7,3),(2,5,8),(2,6,3)...

 So there are 70 triples in this system.

**52**

 (1)Let L be an 3-by-6 Latin rectangle based on Z6.Define a bigraph G =(X, △, Y), X = {x0, x1, …, x5} corresponds to columns 0, 1, …, 5 of the rectangle L, Y = {0, 1, …, 5} is the elements on which L is based. △ = {(xi, j): j does not occur in column i of L}. G has a perfect matching. Suppose the edges of a perfect matching are{(x0, i0), (x1, i1), …., (x5, i5)}. Then 4-by-6 array obtained by adjoining i0, i1, …,i5 as a new row is a Latin rectangle. Continue the process until the 6-by-6 Latin square is completed.





![daa06ddc-a041-40bb-b3ec-b7401bc1fe94](C:\Users\王乐翔\Desktop\combition_math\hw\daa06ddc-a041-40bb-b3ec-b7401bc1fe94.png)

| 0    | 1    | 2    | 3    | 4    | 5    |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 4    | 3    | 1    | 5    | 2    | 0    |
| 5    | 4    | 3    | 0    | 1    | 2    |
| 1    | 2    | 0    | 4    | 5    | 3    |
|      |      |      |      |      |      |
|      |      |      |      |      |      |

![78cf1643-22ce-4362-bb1b-f2edd0c43313](C:\Users\王乐翔\Desktop\combition_math\hw\78cf1643-22ce-4362-bb1b-f2edd0c43313.png)

| 0    | 1    | 2    | 3    | 4    | 5    |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 4    | 3    | 1    | 5    | 2    | 0    |
| 5    | 4    | 3    | 0    | 1    | 2    |
| 1    | 2    | 0    | 4    | 5    | 3    |
| 2    | 0    | 5    | 1    | 3    | 4    |
|      |      |      |      |      |      |

| 0    | 1    | 2    | 3    | 4    | 5    |
| ---- | ---- | ---- | ---- | ---- | ---- |
| 4    | 3    | 1    | 5    | 2    | 0    |
| 5    | 4    | 3    | 0    | 1    | 2    |
| 1    | 2    | 0    | 4    | 5    | 3    |
| 2    | 0    | 5    | 1    | 3    | 4    |
| 3    | 5    | 4    | 2    | 0    | 1    |

**56**

 (1)Let L be a semi-Latin square of order 7 and index 4. 

Define a bigraph G =(X, △, Y), X = {x0, x1, …, x6} correspond to rows 0, 1, …, 6 of the rectangle L, Y = {y0, y1, …, y6} correspond to columns of L. △ = {(xi, yj): the position at row i column j is unoccupied. Then G is 3-regular and has a perfect matching. This matching identifies the desired position for number 4. Continue to place other numbers 5, 6…. until L is completed.

![3](C:\Users\王乐翔\Desktop\combition_math\hw\3.png)

| 0    | 2    | 1    | 4    |      |      | 3    |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 2    | 0    | 4    | 1    |      | 3    |      |
| 3    |      | 0    | 2    | 1    |      | 4    |
|      | 3    | 2    | 0    | 4    | 1    |      |
| 4    |      | 3    |      | 0    | 2    | 1    |
| 1    | 4    |      |      | 3    | 0    | 2    |
|      | 1    |      | 3    | 0    | 4    | 0    |

![4](C:\Users\王乐翔\Desktop\combition_math\hw\4.png)

| 0    | 2    | 1    | 4    | 5    |      | 3    |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 2    | 0    | 4    | 1    |      | 3    | 5    |
| 3    |      | 0    | 2    | 1    | 5    | 4    |
| 5    | 3    | 2    | 0    | 4    | 1    |      |
| 4    | 5    | 3    |      | 0    | 2    | 1    |
| 1    | 4    |      | 5    | 3    | 0    | 2    |
|      | 1    | 5    | 3    | 0    | 4    | 0    |

| 0    | 2    | 1    | 4    | 5    | 6    | 3    |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 2    | 0    | 4    | 1    | 6    | 3    | 5    |
| 3    | 6    | 0    | 2    | 1    | 5    | 4    |
| 5    | 3    | 2    | 0    | 4    | 1    | 6    |
| 4    | 5    | 3    | 6    | 0    | 2    | 1    |
| 1    | 4    | 6    | 5    | 3    | 0    | 2    |
| 6    | 1    | 5    | 3    | 0    | 4    | 0    |