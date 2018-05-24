# 协同过滤
若以$M$表示$m$行，$n$列的用户-项目的历史评分矩阵，以$\overrightarrow{u}$表示待产生推荐结果的目标用户，近邻用户群尺寸为$k$，则基于用户的协同过滤推荐系统的核心实现可以用以下伪代码表示：

$similarity\leftarrow \left [ s_{1},s_{2}\cdots s_{n} \right ]$

for $\overrightarrow{m_{i}} \leftarrow M$ do

$s_{i} \leftarrow \rho (\overrightarrow{u},\overrightarrow{m_{i}})$

endfor

$sorted \leftarrow sort(similarity)$

$most \leftarrow$取$similarity$前k个元素

$s\leftarrow 0,s_{w} \leftarrow \left [ 0,0,\cdots ,0 \right ]$

for $s{i}\leftarrow similarity$ do

$s_{w}\leftarrow s_{w}+(s_{i}\ast m_{i})$

$s\leftarrow s+s_{i}$

endfor

$p\leftarrow \frac{s_{w}}{s}$

# 局部敏感哈希

## 离线生成索引
若以$M$表示$m$行，$n$列的用户-项目的历史评分矩阵，$T$表示尺寸为$t$的哈希表组，则离线生成$r$维索引的过程可以用以下伪代码表示：

$T\leftarrow$尺寸为$t$的空哈希表组

for $i\leftarrow$ 1 to $t$ do

$R\leftarrow$哈希表$t_{i}$对应的$r$个$m$维随机向量

for $u_{j}\leftarrow M$ do

$index\leftarrow 0$

for $r_{k}\leftarrow R$ do

index << 1

if $u_{j}\cdot r{k}\leq 0$ do

$index\leftarrow index|0$

else

$index\leftarrow index|1$

$T_{i,index}\leftarrow u_{j}$

endif

endfor

endfor

endfor

## 在线查找相似用户

若以$S$表示返回的近邻相似用户集合，$u_{t}$表示待求近邻相似用户群的目标用户，则此查找过程可以用以下伪代码表示：

$S\leftarrow \left \langle  \right \rangle$

for $i\leftarrow$ 1 to $t$ do

$R\leftarrow$哈希表$t_{i}$对应的$r$个$m$维随机向量

for $u_{j}\leftarrow M$ do

$index\leftarrow 0$

for $r_{k}\leftarrow R$ do

index << 1

if $u_{j}\cdot r{k}\leq 0$ do

$index\leftarrow index|0$

else

$index\leftarrow index|1$

$S\leftarrow$哈希表$T_{i,index}$中的的所有用户评分向量

endif

endfor

endfor

endfor

# 同态加密

我们用角标$p$表示变量值为明文(Plain Text)，角标$c$表示变量值为密文(Cipher Text)。整个过程可以用以下伪代码表示：

$k_{public},k_{private}\leftarrow$生成Paillier密钥对

$u_{c}\leftarrow E(k_{public},u_{p})$

$m_{c},n_{c}\leftarrow$计算$(u_{pi}-\bar u_{p})$及其开平方结果并加密

$s,s_{w}\leftarrow$使用自身数据基于协同过滤计算结果

$index\leftarrow$计算局部敏感哈希表索引

for $p\leftarrow$参与计算的平台列表 do

$x,y\leftarrow Participate(p,u_{c},index,m_{c},n_{c})$

endfor

$s_{w}\leftarrow s_{w}+D(k_{private},x)$

$s\leftarrow s+D(k_{private},y)$

$result\leftarrow \frac{s_{w}}{s}$

其中，$Participate(p,u_{c},index,m_{c},n_{c})$为请求其他平台参与计算的函数，参数的含义依次为参与的平台，加密后的用户评分向量，用户向量在局部敏感哈希表组中的索引情况，加密后的用户评分向量标准化计算结果，加密后的用户评分向量标准化并开平方的计算结果。角标$c$表明其值为密文。其实现可以用以下伪代码表示：

${M}'\leftarrow$从$LSH(index)$中随机选择$k$个

$S\leftarrow \left [ s_{1c},s_{2c}\cdots s_{kc} \right ]$

for $u_{i}\leftarrow {M}'$ do

$s_{ic}\leftarrow \rho_{paillier}(p,u_{c},m_{c},n_{c})$

endfor

$s_{c}\leftarrow 0,s_{wc} \leftarrow \left [ 0,0,\cdots ,0 \right ]$

for $i\leftarrow$ 1 to k do

$s_{c}\leftarrow s_{c}+s_{ic}$

$s_{wc}\leftarrow s_{wc}+s_{ic}\ast m_{i}$

endfor

return $s_{wc},s_{c}$
