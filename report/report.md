# CS205 Report#4
**姓名：** 刘啸涵
**SID：** 11911925

## Part1 问题描述
- 本项目旨在利用SIMD，MetaOperation等工具实现快速的矩阵乘法。给定的输入包含大尺寸（可能高达64K×64K）的矩阵；矩阵元素为随机生成。由于所有的矩阵都是随机生成，元素等于0的概率接近于0。在这种情况下，矩阵必然是稠密的。这意味着很多针对稀疏矩阵的优化算法并不可用。


    