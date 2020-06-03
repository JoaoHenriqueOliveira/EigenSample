## EigenSample
Python module for performing data augmentation
The creation of this module is based on the article below:
_____________________________________________________________________________________________________________________________ 

Jayadeva, Sumit Soman, Soumya Saxena,

****EigenSample: A non-iterative technique for adding samples to small datasets****,

Applied Soft Computing,\

Volume 70,\

2018,\

Pages 1064-1077,\

ISSN 1568-4946,\

[Science Direct](http://www.sciencedirect.com/science/article/pii/S1568494617304994)\

Abstract: In many engineering applications, experimental data is scant and tedious to generate. When performing a preliminary analysis of such data, the analyst is often challenged by the small size of the data set. An alternate, widely used approach to address this is bootstrapping, where synthetic samples are generated by randomly sampling the available dataset. In this paper, we suggest a way to augment small data sets in a manner that least distorts the original distributions. We do this by first projecting the dataset into the subspace spanned by the first k principal components, where k is appropriately chosen; next, additional samples are generated in the lower dimensional space. These additional samples are then projected back into the original space. The focus of this paper is on the inverse projection task, which arises in many other contexts. Well known solutions include using the Moore-Penrose inverse, which yields a minimum norm solution; an alternative is to solve a linear system of equations. However, these solutions do not typically preserve bounds on the input data. We propose a solution that allows a trade-off between minimizing the norm of the solution and the approximation error involved in the projection. We further develop a least-squares version of this approach in order to obtain a non-iterative procedure to obtain the inverse projection. We name our approach EigenSample, since additional samples are added in the eigenspace of the original dataset. The performance of our proposed approach is compared with three competing approaches for augmenting datasets and evaluated using multiple classification algorithms; and the results indicate the superior performance of our proposed method.
