Paillier Example
================

同态密码
------------

同态，通俗理解就是运算方可以在不知道运算数明文的情况下对密文进行某种计算，使得计算的结果仍为合法的密文，且解密后恰好等于原两明文的某种计算的结果。

假设加密算法为 :math:`E(x)`，其中 :math:`x` 为明文，则同态密码系统可以表示为符合以下性质：

.. math::

  E(x) \oplus E(y) = E(x \circ y)

其中，:math:`x, y` 为明文， :math:`\oplus` 是密文上的某种运算，:math:`\circ` 是明文上的某种运算，通常为算数或逻辑的 *加法或乘法* 。

当一个同态密码系统同时支持明文上的加法和乘法，即以下二式成立：

.. math::

  \begin{aligned}
    E(x) \oplus E(y) = & E(x + y) \\
    E(x) \otimes E(y) = & E(xy)
  \end{aligned}

则称该系统为 **全同态密码系统** ，若仅支持加法或乘法中的一个，则称为 **半同态密码系统** 。

Paillier 半同态密码系统
----------------------------

Paillier 同态密码系统是由 Pascal Paillier 在 1999 年提出的半同态密码系统。它以密文乘法的形式计算明文上的同态模加法，因此它是一个半同态密码系统。

我们将在接下来的部分中介绍一个简单的 python 版本的 Paillier 密码系统的实现，并在最后证明其正确性。

Paillier 密码系统的简单实现
----------------------------------

数学基础
^^^^^^^^^^^^^^^

首先我们不加解释地实现带模快速幂、拓展欧几里得算法，并提供求模意义下乘法逆元的实现。

.. code-block:: python
  
  import random
  import math

  # quick power: calculate (base^exponent)%modulus
  def powerMod(base, exponent, modulus):
    answer = 1
    while exponent > 0:
      if exponent % 2 == 1: answer = (answer * base) % modulus
      base = (base**2) % modulus
      exponent //= 2
    return answer

  # exgcd: return (x, y, gcd(a, b)) where ax + by = gcd(a,b)
  def exgcd(a, b):
    if b == 0: return 1, 0, a
    else:
      x0, y0, g = exgcd(b, a%b)
      return y0, x0 - (a//b)*y0, g

  # inv: return x, where ax = 1 (mod m)
  def inv(a, m) -> int:
      x, y, g = exgcd(a, m)
      return (x%m+m)%m

由于密钥生成算法需要使用大质数，为了简单起见，我们使用埃拉托色尼筛法。

.. code-block:: python

  def sieve(upperbound = 0x4000):
    primes = []
    flags = [True] * upperbound
    for each in range(2, upperbound):
      if not flags[each]: continue
      for multiplier in range(2, upperbound // each):
        flags[multiplier * each] = False
      primes.append(each)
    return primes

  primes = sieve(0x4000)

.. note::
  在实际应用中，为了快速找出足够大的质数，通常使用 Miller Rabin 素性检测的方法。该算法本质是随机生成一个数字，若它通过简单的几步检测，即可以较高的概率认为它是一个质数。

密钥生成
^^^^^^^^^^^^^^^^^^^^

Paillier 密码系统的密钥生成步骤如下：

#. 选取两个随机大素数 :math:`p, q` ，计算 :math:`n=pq, \lambda=\text{lcm}(p-1, q-1)` ；
#. 选取 :math:`g < n^2` ，使得 :math:`g` 与 :math:`n` 互质，且模 :math:`n` 意义下的乘法逆元

   .. math::

     \mu \cdot L(g^\lambda \text{ mod } n^2) \equiv 1 \pmod {n}

   存在，其中 :math:`L(x) = (x-1)/n` ；

#. 公钥为 :math:`(n, g)` ，私钥为 :math:`(\lambda, \mu)` 。

.. note::
  
  从实现的角度而言，因为 :math:`g` 是公钥，所以不必选取 :math:`g` 为随机数，例如可以直接选取 :math:`g = n+1` 。

.. code:: python

  # produce (n, g, lambda, mu), where (n, g) is the public key, (lambda, mu) is the private key
  def generateKeys():
    primeCount = len(primes)
    p = primes[random.randint(primeCount // 2, primeCount)]
    while True:
      q = primes[random.randint(primeCount // 2, primeCount)]
      if p != q: break
    n = p*q
    Lambda = (p-1)*(q-1) // math.gcd(p-1, q-1)
    g = n + 1
    mu = inv((powerMod(g, Lambda, n*n)-1)//n, n)
    return n, g, Lambda, mu

加密算法
^^^^^^^^^^^^^^^^^^^^^^^

Paillier 密码系统的加密步骤：对于明文 :math:`m < n` 随机选取 :math:`0 < r < n` 使得 :math:`r` 与 :math:`n` 互质，则密文为 :math:`c = g^m r^n (\text{mod } n^2)` 。

.. note::
  实际上，当 :math:`n` 足够大时，可以直接随机选取 :math:`0 < r < n` ，因为二者不互质的概率极小。

.. code:: python

  def encrypt(m, n, g):
    while True:
      r = random.randint(1, n-1)
      if math.gcd(r, n) == 1: break
    c = powerMod(g, m, n*n) * powerMod(r, n, n*n) % (n*n)
    return c

解密算法
^^^^^^^^^^^^^^^^^^^^^^^^^

Paillier 密码系统的解密步骤：对于密文 :math:`c` ，明文为 :math:`m = \mu \cdot L(c^\lambda \text{ mod } n^2) \text{ mod } n` 。

.. code:: python

  def decrypt(c, Lambda, mu, n):
    k = powerMod(c, Lambda, n*n)
    assert((k-1)%n == 0) # when (k-1)%n != 0, c is not a valid ciphertext.
    return (k-1)//n * mu % n  

