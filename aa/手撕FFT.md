# 手撕FFT

## 多项式简介

算法导论提供了全部理论基础：

先说分治：

我们在相乘时，按照未知项的奇偶性分开：

$A(x) = A^0(x) + A^1(x) $;

$B(x) = B^0(x) + B^1(x)$;

$AB = (A^0 + xA^1)(B^0 + xB^1) = A^0B^0 + x(A^1B^0 + A^0B^1) + x^2A^1B^1$;

由上式可得，我们可以通过分治算法把两个多项式折半，再计算四次多项式乘法并相加合并。

但此时$T(n) = 4T(n/2) + f(n)$，所以复杂度仍为$O(n^2)$;

但是$(ax + b)(cx + d) = acx^2 + (ad + bc)x + bd$，实际上只需要三次乘法就可以，所以我们可以使用这个方法减少一次乘法运算，此时$T(n) = 3T(n/2) + f(n)$;

我们得知多项式可以使用点值表示和插值表示两种形式；

我们使用拉格朗日插值求解方法可以将复杂度优化到$n^2$：

- 选取$n$个$x^i$，带入点值，复杂度为$O(n^2)$;
- 计算点值的卷积，复杂度为$O(n)$;
- 插值计算系数向量，这一步是$O(n^2)$;

我们在此基础上通过选取复数单位根继续优化：

- 考虑方程$z^n = 1$，因此在一个三角函数周期上取得n个方程复数根；
- 相消定理，其实就是周期函数，为了限制右上角次数；
- 折半定理，n次单位根的平方集合等于n/2次单位根的集合，显然成立，得到结论；
- 求和引理，就是凑够了就是0；

再说DFT：

DFT就是将次数界为n的多项式A(x)在n次单位复数根上求值的过程；

$y = DFT(a)$

因此我们使用FFT利用单位根的特殊性质把DFT优化到$O(nlogn)$:

- 在分治中我们要计算的是$A^0(x^2)$，根据折半定理$(\omega^0)^2...(\omega^k)^2...$，两两重复，所以是n/2个n/2次单位根；
- 然后合并答案：计算只需$yi = yi^0 + \omega^iyi^1, y(i + n/2) = yi^0 - \omega^iyi^i$;
- $T(n) = 2T(n/2) + f(n), O(nlogn)$；

因为按照奇偶性计算，所以使用蝴蝶操作，将所有系数按照位置排列再迭代合并。

### 位反转排序

```cpp
for(int i = 0; i < n; i++){
    rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
    if(i < rev[i]){
        swap(A[i], A[rev[i]]);
    }
}
```

- **位反转数组**：使用位操作计算`rev[i]`，将索引`i`的二进制表示进行反转。
- **交换**：如果`i`小于`rev[i]`，则交换`A[i]`和`A[rev[i]]`，实现数组的位反转排序。这是FFT算法中的关键步骤，有助于提高计算效率。

### 例子：位反转排序

假设我们有一个数组的长度为8（n=8n = 8n=8），其索引为0到7。我们的目标是将这些索引进行位反转。

#### 1. 原始索引及其二进制表示

```
索引:   0   1   2   3   4   5   6   7
二进制: 000 001 010 011 100 101 110 111
```

#### 2. 位反转过程

对于每个索引，我们将其二进制表示进行反转：

- `0` -> `000` -> `000` -> `0`
- `1` -> `001` -> `100` -> `4`
- `2` -> `010` -> `010` -> `2`
- `3` -> `011` -> `110` -> `6`
- `4` -> `100` -> `001` -> `1`
- `5` -> `101` -> `101` -> `5`
- `6` -> `110` -> `011` -> `3`
- `7` -> `111` -> `111` -> `7`

#### 3. 反转结果

反转后的索引数组是：

```
索引:   0   4   2   6   1   5   3   7
```

### 应用位反转排序的FFT

假设我们有一个复数数组 AAA：

```
A: [A[0], A[1], A[2], A[3], A[4], A[5], A[6], A[7]]
```

经过位反转排序后，数组会变为：

```
A: [A[0], A[4], A[2], A[6], A[1], A[5], A[3], A[7]]
```

### 蝶形计算的基本形式

对于输入的两个复数 xxx 和 yyy，蝶形计算可以表示为：

$输出1=x+ω⋅y$

$输出2=x−ω⋅y$

其中，$\omega$ 是旋转因子，通常是一个复数，表示特定的相位旋转，依赖于当前的计算阶段。

### 内循环进行蝶形运算

```cpp
for(int i = 0; i < n; i += mid << 1){
```

- `i`循环遍历`A`，每次跳过`mid << 1`（即`2 * mid`），这保证了在进行蝶形运算时不会重叠。

### 计算蝶形操作

```cpp
for(int j = 0; j < mid; j++, omega *= temp){
```

- 内部循环用于进行蝶形操作，`j`从0到`mid-1`，更新`omega`为当前的旋转因子。

```cpp
complex<double>x = A[i + j], y = omega * A[i + j + mid];
```

- 取出当前需要计算的两个元素，`x`为前半部分，`y`为后半部分乘以旋转因子。

```cpp
A[i + j] = x + y;
A[i + j + mid] = x - y;
```

- 更新数组A的值：
  - `A[i + j]`存储前半部分和后半部分的和（频域的合成）。
  - `A[i + j + mid]`存储前半部分和后半部分的差（频域的分离）。



### 函数 `invert`

```cpp
int invert(int n){
    int bit = 1;
    while((1 << bit) < n) bit++;
    return (1 << bit);
}
```

- 该函数返回大于等于`n`的最小的2的幂次。
- 通过位运算计算出2的幂次，确保FFT算法能够处理的长度是2的幂次。

### 函数 `FFT`

```cpp
void FFT(complex<double> *A, int n, int inv){
    int bit = 1;
    while((1 << bit) < n) bit++;
    for(int i = 0; i < n; i++){
        rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
        if(i < rev[i]){
            swap(A[i], A[rev[i]]);
        }
    }
    
    for(int mid = 1; mid < n; mid <<= 1){
        complex<double> temp(cos(Pi / mid), inv * sin(Pi / mid));
        for(int i = 0; i < n; i += mid << 1){
            complex<double> omega(1, 0);
            for(int j = 0; j < mid; j++, omega *= temp){
                complex<double>x = A[i + j], y = omega * A[i + j + mid];
                A[i + j] = x + y;
                A[i + j + mid] = x - y;
            }
        }
    }
}
```

- 参数：

  - `A`：输入的复数数组。
  - `n`：数组长度。
  - `inv`：指示是进行正向FFT还是逆向FFT（`1`表示正向，`-1`表示逆向）。

- 功能：

  1. 计算并存储`rev`数组，用于位反转。
  2. 使用蝶形操作对复数进行FFT计算。`temp`是旋转因子，根据当前的`mid`值计算出。
  3. 通过循环进行合并和计算，最终得到频域结果。

  

```C
#include <cstdio>
#include <complex>
using namespace std;
const int N = 1e7 + 1;
const double Pi = acos(-1);
int n, m, rev[N];
complex<double> F[N], G[N], H[N];

int invert(int n){
	int bit = 1;
	while((1 << bit) < n)bit++;
	return (1 << bit);
}

int getint(){
	int x = 0, f = 1; char c = getchar();
	while(c < '0' || c > '9'){
		if(c == '-')f = -1;
		c = getchar();
	}
	while(c >= '0' && c <= '9'){
		x = (x << 1) + (x << 3) + c - '0';
		c = getchar();
	}
	return x * f;
}

void FFT(complex<double> *A, int n, int inv){
	int bit = 1;
	while((1 << bit) < n)bit++;
	for(int i = 0; i < n; i++){
		rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (bit - 1));
		if(i < rev[i]){
			swap(A[i], A[rev[i]]);
		}
	}
	
	for(int mid = 1; mid < n; mid <<= 1){
		complex<double> temp(cos(Pi / mid), inv * sin(Pi / mid));
		for(int i = 0; i < n; i += mid << 1){
			complex<double> omega(1, 0);
			for(int j = 0; j < mid; j++, omega *= temp){
				complex<double>x = A[i + j], y = omega * A[i + j + mid];
				A[i + j] = x + y;
				A[i + j + mid] = x - y;
 			}
		}
	}
}

int main(){
	scanf("%d %d", &n, &m);
	for(int i = 0; i <= n; i++)F[i].real(getint());
	for(int i = 0; i <= m; i++)G[i].real(getint());
	//printf("get done\n");
	FFT(F, invert(n + m), 1);
	FFT(G, invert(n + m), 1);
	
	for(int i = 0; i <= invert(n + m); i++){
		H[i] = F[i] * G[i];
	}
	
	FFT(H, invert(n + m), -1);
	
	for(int i = 0; i <= n + m; i++){
		printf("%d ", (int)(H[i].real() / invert(n + m) + 0.5));
	}
}

```

