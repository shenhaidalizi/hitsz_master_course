# HW1

1. Show that if **n**+1 distinct integers are chosen from the set {1, 2, …, 3**n**}, then there are always two which differ by at most 2.

Solution: We divide the set into n sets, while puting the first closed three numbers into the same number from head to tail. Then we get n sets, we choose n + 1 integers from sets, so there are always choosen two integers from the same small set under the Pigeonhole Principle, for we choose n + 1 integers from n sets. So the two integers from the same set must differ at most 2. 

2. **Prove that of any five points chosen within a square of side length 1, there are two whose distance apart is at most sqrt(1/2)**  **.** 

Solution:  We can divide the square into four parts. We can see that if we put the points into a small part, in this part the most distance is sqrt(1/2). So if there are two points can be put into one part, there are always two points whose distance apart is at most sqrt(1/2). 

We prove this problem from the opposition. If there is no point in the same part, we can put the first four points into the four parts. But the fifth point must be put into one part, so there is alway can be proved that there are at least two points put into the same part. So there are always two whose distance apart is at most sqrt(1/2).

3.**In a room there are 10 people with integer ages [1, 60]. Prove that we can always find two groups of people (with no common person) the sum of whose ages is the same.**

Solution: If we divide the 10 people into two groups, there is 2^(10) functions, but we can not divide it into empty set, so there are still 1022 solutions.  For a group, the sum of the ages ranges from 10 to 600, so there is 591 possibilities for the age number. We have 1022 non-empty subsets but only 591 possibilities, so under the Pigeonhole Prinsiple, at least two of these subsets must sum to the same value.