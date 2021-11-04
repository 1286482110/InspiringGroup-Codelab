pinocchio-based-zksnarks
========================

李琪 Li Qi

.. _1-零知识证明和zksnarks:

1 零知识证明和zksnarks
----------------------

.. _11-为什么需要可证明的计算:

1.1 为什么需要可证明的计算？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

随着技术的发展，计算能力表现出不对称的特性，例如云计算等拥有大量的算力，而移动设备等算力十分有限。因此，一些计算能力较弱的客户端设备希望通过外包计算的方式，将计算任务外包给算力强大的设备。而此时，这些客户端就希望在得到计算结果的同时，可以验证结果的正确性，以防止偶然的错误或恶意的攻击。同时，从另一方面，提供外包计算服务的服务商也希望可以证明自己的工作，这样，他们既可以要求更高的价格，又可以摆脱不必要的责任。

.. _12-什么是零知识证明:

1.2 什么是零知识证明?
~~~~~~~~~~~~~~~~~~~~~

零知识证明是指证明者能够在不向验证者提供任何有用的信息的情况下，使验证者相信某个论断是正确的。

一个简单的例子：A要向B证明自己拥有某个房间的钥匙（通常情况下，我们把A称为证明者Prover，B称为验证者Verifier），假设该房间只能用钥匙打开锁，其他任何方法都打不开，而且B确定该房间内有某一物体。此时A用自己拥有的钥匙打开该房间的门，然后把物体拿出来出示给B，从而证明自己确实拥有该房间的钥匙。\ **在这个过程中，证明者A能够在不给验证者B看到钥匙的情况下，使B相信他是有钥匙开门的。**

我们可以通过零知识证明的思路，实现可证明的计算。

.. _13-什么是zksnarks:

1.3 什么是zksnarks?
~~~~~~~~~~~~~~~~~~~

zk-SNARK（ Zero-Knowledge Succinct Non-Interactive Argument of
Knowledge）是零知识证明的一种形式，它只适用于满足QAP(Quadratic
Arithmetic Programs)形式的计算问题。zksnarks具有以下性质：

简明 (Succinctly) : 独立于计算量，证明是恒定的，小尺寸的。

非交互性 (Non-interactive) : 证明只要一经计算就可以在不直接与 prover
交互的前提下使任意数量的 verifier 确信。

可论证的知识 (Argument of Knowledge)
:对于陈述是正确的这点有不可忽略的概率，即无法构造假证据；并且 prover
知道正确陈述的对应值（即：证据）。

零知识( zero-knowledge) :
很难从证明中提取任何知识，即它与随机数无法区分。

下面给出一种简单的zksnarks协议：匹诺曹协议（\ `Pinocchio
protocol <https://eprint.iacr.org/2013/279.pdf>`__\ ）。

.. _14-什么是qapquadratic-arithmetic-program）:

1.4\* 什么是QAP（Quadratic Arithmetic Program）？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

QAP（\ `Quadratic Arithmetic
Program <https://link.springer.com/content/pdf/10.1007/978-3-642-38348-9_37.pdf>`__\ ）的定义：

域\ :math:`F`\ 上的QAP
:math:`Q`\ 包含三组\ :math:`m+1`\ 多项式\ :math:`V=\{v_k(x)\}`\ ，\ :math:`W=\{w_k(x)\}`\ ，\ :math:`Y=\{y_k(x)\}`\ （其中\ :math:`k \in \{0...m\}`\ ）和一个目标多项式\ :math:`t(x)`\ 。假设\ :math:`f`\ 是一个函数，它以域\ :math:`F`\ 上的
:math:`n` 个元素作为输入，\ :math:`n'`\ 个元素作为输出，总共有
:math:`N = n + n'`\ 个输入输出元素。

此时，我们可以说，使用Q计算f，如果\ :math:`(c_1,...c_N)\in F^N`\ 是 f
的输入和输出的有效赋值，当且仅当存在系数\ :math:`(c_1,...c_N)` 使 t
(x)整除 p (x)，其中

:math:`p(x) = (v_0(x)+\Sigma_{k=1}^m{c_k \cdot v_k(x)}) \cdot (w_0(x)+\Sigma_{k=1}^m{c_k \cdot w_k(x)}) \cdot (y_0(x)+\Sigma_{k=1}^m{c_k \cdot y_k(x)})`

换句话说，一定存在多项式\ :math:`h(x)`\ 使得\ :math:`h(x)t(x)=p(x)`\ 。\ :math:`Q`
的大小是\ :math:`m`\ ，\ :math:`Q`\ 的阶数等于\ :math:`t(x)`\ 的阶数。

我们通常使用拉格朗日插值法构建QAP。

.. _2-匹诺曹协议pinocchio-protocol）:

2 匹诺曹协议（Pinocchio protocol）
----------------------------------

.. _21-传统的公开可证明计算public-verifiable-computation）流程:

2.1 传统的公开可证明计算（Public Verifiable Computation）流程
-------------------------------------------------------------

公开可证明计算一般分为以下三步：

1. 验证者（verifier）初始化：\ :math:`(EK_F,VK_F)\leftarrow KeyGen(F,1^{\lambda})`

   验证者需要证明者使用私有数据\ :math:`u`\ 计算函数\ :math:`F`\ ，则他首先为函数\ :math:`F`\ 生成公开密钥\ :math:`EK`\ （public
   evaluation key）和验证密钥\ :math:`VK`\ （public verification key
   ）。将函数\ :math:`F`\ 和公开密钥\ :math:`PK`\ 发给证明者。【注：\ :math:`VK`\ 也是公开的，任何人都可以使用\ :math:`VK`\ 验证计算结果的正确性。】

2. 证明者（prover）提供证明：\ :math:`(y,\pi_y) \leftarrow Compute(EK_F;u)`

   证明者拿到函数\ :math:`F`\ 之后，使用自己的私有数据\ :math:`u`\ 进行计算，得到输出\ :math:`y`\ 。同时，根据计算的中间结果，生存证明\ :math:`\pi`\ 。将计算结果\ :math:`y`\ 和证明\ :math:`\pi`
   发给验证者。

3. 验证者（verifier）验证证明：\ :math:`\{0,1\}\leftarrow Verify(VK_F,y,\pi_y)`

   验证者拿到计算结果\ :math:`y`\ 和证明\ :math:`\pi`\ 后，使用验证密钥\ :math:`VK`\ 对结果进行验证。

.. _22-匹诺曹协议pinocchio-protocol）:

2.2 匹诺曹协议（Pinocchio protocol）
------------------------------------

匹诺曹协议可以理解为Public Verifiable Computation的一种实现方法。

1. 验证者（verifier）初始化：\ :math:`(EK_F,VK_F)\leftarrow KeyGen(F,1^{\lambda})`

   设函数\ :math:`F`\ 共有\ :math:`N`\ 个输入/输出。首先将\ :math:`F`\ 转换为一个算数电路\ :math:`C`\ ，然后将\ :math:`C`\ 编译成为对应的QAP\ :math:`Q=((t_x),V,W,Y)`
   ，Q的大小为m阶数为d。\ :math:`I_{mid} = \{N+1,...,m\}`\ 是输入输出无关值（non-IO-related
   indices），e是一个非平凡的双线性映射（non-trivial bilinear
   map）\ :math:`e:G\times G \rightarrow G_T`\ 。g是G的生成元。

   选择随机数\ :math:`r_v,r_w,s,\alpha_v,\alpha_w,\alpha_y,\beta,\gamma \stackrel{R}{\leftarrow} F`\ 。设\ :math:`r_y = r_v \cdot r_w, g_v = g^{r_v}, g_w = g^{r_w}, g_y = g^{r_y}`\ 。
   
   .. math::
   
     \begin{aligned}
       EK_F = \{g_v^{v_k(s)}\}_{k \in I_{mid}},\{g_w^{w_k(s)}\}_{k \in I_{mid}},\{g_y^{y_k(s)}\}_{k \in I_{mid}},\{g_v^{\alpha_v v_k(s)}\}_{k \in I_{mid}},\{g_w^{\alpha_w w_k(s)}\}_{k \in I_{mid}},\\
       \{g_y^{\alpha_yy_k(s)}\}_{k \in I_{mid}}, \{g^{s^i}\}_i \in [d],\{g_v^{\beta v_k(s)} g_w^{\beta w_k(s)} g_y^{\beta y_k(s)}\}
     \end{aligned}
   
   .. math::

     VK_F = (g^1,g^{\alpha_v},g^{\alpha_w},g^{\alpha_y}, g^{\gamma},g_y^{t(s)},\{g_v^{v_k(s)},g_w^{w_k(s)},g_y^{y_k(s)}\}_{k\in{0}\cup[N]}

2. 证明者（prover）提供证明 :
   :math:`(y,\pi_y) \leftarrow Compute(EK_F;u)`

   对于输入 u ，证明者计算得到 f (u) ;
   同时他还得到了中间变量\ :math:`\{c_i\}_{i\in[m]}`\ 。他得到了h
   (x)通过p (x) = h (x) t (x)，并计算了证明\ :math:`\pi`\ 。

   :math:`\pi = (g_v^{v_{mid}(s)},g_w^{w_{mid}(s)},g_y^{y_{mid}(s)},g^{h(s)},g_v^{\alpha_vv_{mid}(s)},g_w^{\alpha_ww_{mid}(s)},g_y^{\alpha_yy_{mid}(s)},g_v^{\beta v_{mid}(s)}g_w^{\beta w_{mid}(s)}g_y^{\beta y_{mid}(s)})` 

      

   其中，\ :math:`v_{mid}(x) = \Sigma_{k \in I_{mid}}c_k \cdot v_k(s)`\ ，同理计算\ :math:`w_{mid}(s),y_{mid}(s)`\ 。

3. 验证者（verifier）验证证明 :
   :math:`\{0,1\}\leftarrow Verify(VK_F,y,\pi_y)`

   将证明\ :math:`\pi`\ 映射为\ :math:`(g^{V_{mid}},g^{W_{mid}},g^{Y_{mid}},g^H,g^{V'_{mid}},g^{W'_{mid}},g^{Y'_{mid}},g^Z)`\ 。

   使用\ :math:`VK`\ 计算\ :math:`g_v^{v_{io}(s)} = \Pi_{k \in [N]}(g_v^{v_k(s)})^{c_k}`\ ，同理计算\ :math:`g_w^{w_{io}(s)},g_y^{y_{io}(s)}`\ 。

   验证计算正确性：

   :math:`e(g_v^{v_0(s)}g_v^{v_{io}(s)}g_v^{V_{mid}},g_w^{w_0(s)}g_w^{w_{io}(s)}g_w^{W_{mid}}) = e(g_y^{t(s)},g^H)e(g_y^{y_0(s)}g_y^{y_{io}(s)}g_y^{Y_{mid}},g)` 

   验证可变多项式约束：
   
   :math:`e(g_v^{V'_{mid}},g) = e(g_v^{V_{mid}},g^{\alpha_v})` 
   :math:`e(g_w^{W'_{mid}},g) = e(g_w^{W_{mid}},g^{\alpha_w})` 
   :math:`e(g_y^{Y'_{mid}},g) = e(g_y^{Y_{mid}},g^{\alpha_y})` 
      
   验证每个线性组合是否使用了相同的系数：

   :math:`e(g^Z,g^\gamma) = e(g_v^{V_{mid}}g_w^{W_{mid}}g_y^{Y_{mid}},g^{\beta\gamma})`

.. _3-implementation:

3 Implementation
----------------

匹诺曹协议的实现方法参考
`Go-snark <https://github.com/shamatar/go-snarks.git>`__\ 和\ `go-snark-study <https://github.com/arnaucube/go-snark-study>`__\ 。这里使用\ `V神(Vitalik
Buterin)的例子 <https://medium.com/@VitalikButerin/zk-snarks-under-the-hood-b33151a013f6>`__\ 进行实现。完整代码见\ `GitHub <https://github.com/liqi16/pinocchio-protocol-zksnarks.git>`__\ 。代码运行方式：

.. code:: shell

   go get github.com/arnaucube/go-snark
   go get github.com/arnaucube/go-snark/circuitcompiler
   go run main.go

以下代码中的\ :math:`(Pk, Vk)`\ 对应上述公式中的\ :math:`(Ek, Vk)`;\ :math:` (A,B,C)`\ 对应上述公式中的\ :math:`(V,W,Y)`\ 。

实现的总体架构如下：

.. code:: go

   func main() {

   	//verifier初始化
   	flatCode := PrepareCircuit()

   	circuit := CompileCircuit(flatCode)

   	setup := TrustedSetup(circuit)

   	pk := setup.Pk
   	vk := setup.Vk
     
     /*verfier将circuit,pk交给prover*/

   	//prover提供证明
   	inputs := PrepareInputAndOutput()

   	proof := GenerateProofs(circuit, pk, inputs)
     
     /*prover将proof,inputs.Public[35]交给prover*/

   	//verifier验证证明
   	verified := VerifyProofs(vk, inputs.Public, proof)

   	if !verified {
   		fmt.Println("proofs not verified")
   	} else {
   		fmt.Println("Proofs verified")
   	}

   }

.. _31-preparecircuit:

3.1 PrepareCircuit
~~~~~~~~~~~~~~~~~~

我们用到的函数是\ :math:`y=x^3 + x + 5`\ 。将这个函数拍平，转换为“一个等式中最多含有一次乘法的形式”。这样我们就得到了一个拍平的函数。

.. code:: go

   func PrepareCircuit() string {

   	flatCode := `
   	func exp3(private a):
   		b = a * a
   		c = a * b
   		return c

   	func main(private s0, public s1):
   		s3 = exp3(s0)
   		s4 = s3 + s0
   		s5 = s4 + 5
   		equals(s1, s5)
   		out = 1 * 1
   	`
   	return flatCode
   }

.. _32-compilecircuit:

3.2 CompileCircuit
~~~~~~~~~~~~~~~~~~

我们将电路编译，并转换为R1CS。

.. code:: go

   func CompileCircuit(flatCode string) circuitcompiler.Circuit {
   	// parse the code
   	parser := circuitcompiler.NewParser(strings.NewReader(flatCode))
   	circuit, err := parser.Parse()
   	panicErr(err)
   	fmt.Println("circuit", circuit)

   	a, b, c := circuit.GenerateR1CS()
   	fmt.Println("\nR1CS:")
   	fmt.Println("circuit.R1CS.A", a)
   	fmt.Println("circuit.R1CS.B", b)
   	fmt.Println("circuit.R1CS.C", c)

   	return *circuit

   }

输出：

.. code:: 

   R1CS:
   circuit.R1CS.A [[0 0 1 0 0 0 0 0] [0 0 1 0 0 0 0 0] [0 0 1 0 1 0 0 0] [5 0 0 0 0 1 0 0] [0 0 0 0 0 0 1 0] [0 1 0 0 0 0 0 0] [1 0 0 0 0 0 0 0]]
   circuit.R1CS.B [[0 0 1 0 0 0 0 0] [0 0 0 1 0 0 0 0] [1 0 0 0 0 0 0 0] [1 0 0 0 0 0 0 0] [1 0 0 0 0 0 0 0] [1 0 0 0 0 0 0 0] [1 0 0 0 0 0 0 0]]
   circuit.R1CS.C [[0 0 0 1 0 0 0 0] [0 0 0 0 1 0 0 0] [0 0 0 0 0 1 0 0] [0 0 0 0 0 0 1 0] [0 1 0 0 0 0 0 0] [0 0 0 0 0 0 1 0] [0 0 0 0 0 0 0 1]]

.. _33-trustedsetup:

3.3 TrustedSetup
~~~~~~~~~~~~~~~~

根据函数生成公开密钥\ :math:`PK`\ 和验证密钥\ :math:`VK`\ 。

.. code:: go

   func TrustedSetup(circuit circuitcompiler.Circuit) snark.Setup {

   	// R1CS to QAP
   	alphas, betas, gammas, _ := snark.Utils.PF.R1CSToQAP(circuit.R1CS.A, circuit.R1CS.B, circuit.R1CS.C)
   	fmt.Println("QAP")
   	fmt.Println(alphas)
   	fmt.Println(betas)
   	fmt.Println(gammas)

   	// calculate trusted setup
   	setup, err := snark.GenerateTrustedSetup(len(circuit.Signals), circuit, alphas, betas, gammas)
   	panicErr(err)
   	fmt.Println("\nt:", setup.Toxic.T)//私钥，可销毁

   	// remove setup.Toxic
   	var tsetup snark.Setup
   	tsetup.Pk = setup.Pk
   	tsetup.Vk = setup.Vk

   	return tsetup
   }

.. _34-prepareinputandoutput:

3.4 PrepareInputAndOutput
~~~~~~~~~~~~~~~~~~~~~~~~~

输入\ :math:`x=3`\ ，按照函数\ :math:`y=x^3 + x + 5`\ ，输出值为\ :math:`y=35`\ 。

.. code:: go

   func PrepareInputAndOutput() circuitcompiler.Inputs {

   	input := `[
   		3
   	]
   	`

   	output := `[
   		35
   	]
   	`

   	var inputs circuitcompiler.Inputs
   	err := json.Unmarshal([]byte(input), &inputs.Private)
   	panicErr(err)
   	err = json.Unmarshal([]byte(output), &inputs.Public)
   	panicErr(err)

   	return inputs

   }

.. _35-generateproofs:

3.5 GenerateProofs
~~~~~~~~~~~~~~~~~~

.. code:: go

   func GenerateProofs(circuit circuitcompiler.Circuit, pk snark.Pk, inputs circuitcompiler.Inputs) snark.Proof {

   	// calculate wittness
   	witness, err := circuit.CalculateWitness(inputs.Private, inputs.Public)
   	panicErr(err)
   	fmt.Println("\nwitness", witness)

   	// flat code to R1CS
   	a := circuit.R1CS.A
   	b := circuit.R1CS.B
   	c := circuit.R1CS.C
   	// R1CS to QAP
   	alphas, betas, gammas, _ := snark.Utils.PF.R1CSToQAP(a, b, c)
   	_, _, _, px := snark.Utils.PF.CombinePolynomials(witness, alphas, betas, gammas)
   	hx := snark.Utils.PF.DivisorPolynomial(px, pk.Z)

   	fmt.Println(circuit)
   	fmt.Println(pk.G1T)
   	fmt.Println(hx)
   	fmt.Println(witness)
   	proof, err := snark.GenerateProofs(circuit, pk, witness, px)
   	panicErr(err)

   	fmt.Println("\n proofs:")
   	fmt.Println(proof)

   	return proof
   }

.. _36-verifyproofs:

3.6 VerifyProofs
~~~~~~~~~~~~~~~~

.. code:: go

   func VerifyProofs(vk snark.Vk, publicinputs []*big.Int, proof snark.Proof) bool {
   	verified := snark.VerifyProof(vk, proof, publicinputs, true)
   	return verified
   }

输出

.. code:: 

   ✓ e(piA, Va) == e(piA', g2), valid knowledge commitment for A
   ✓ e(Vb, piB) == e(piB', g2), valid knowledge commitment for B
   ✓ e(piC, Vc) == e(piC', g2), valid knowledge commitment for C
   ✓ e(Vkx+piA, piB) == e(piH, Vkz) * e(piC, g2), QAP disibility checked
   ✓ e(Vkx+piA+piC, g2KbetaKgamma) * e(g1KbetaKgamma, piB) == e(piK, g2Kgamma)
