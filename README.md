![img](https://img2018.cnblogs.com/blog/1252882/201905/1252882-20190528152503928-982556190.png)

- # 一.问题建模

    - ## 1.评价指标

        - ### 1.1分类指标

            - #### 二分类

                - ##### 精确率

                    - $ P=TP/(TP+FP) $

                - ##### 召回率

                    - $ R=TP/(TP+FN) $

                - ##### F1值

                    - $2PR/(P+R)$ 

                - ##### P-R曲线

                    - 横轴是召回率0-1,纵轴为对应的召回率下的精确率

                - ##### ROC曲线

                    - 横轴为假正率$FPF=FP/(FP+TN)$,纵轴为真正率$TPR=TP/(TP+FN)$

                - ##### AUC

                    - 直接计算ROC曲线下的面积
                        ​Wilcoxon-Mann-Witney Test:测试任意给一个正类样本和一个负类样本,正类样本的score有多大概率大于负类样本的score.

                    - 两种计算方法：

                        ​	O(M*N)：统计一下所有的M*N个正负样本对中,有多少个组中的正样本中的score大于负样本的socre,然后除以MN,M/N为正/负样本的个数

                        ​	O(M+N)：首先对score从大到小排序，然后令最大score对应的sample 的rank为n，第二大score对应sample的rank为n-1，以此类推。然后把所有的正类样本的rank相加，再减去M-1种两个正样本组合的情况。得到的就是所有的样本中有多少对正类样本的score大于负类样本的score。然后再除以M×N。

                - ##### 对数损失

                    - $-log(P(Y|X))$

            - #### 多分类

                - ##### 准确率

                    - $(TP+TN)/ALL$

                - ##### 错误率

                    - $(FP+FN)/ALL$

        - ### 1.2回归指标

        - ### 1.3聚类指标

        - ### 1.4排序指标

            - ##### 平均准确率MAP

            - ##### NDCG归一化贴现累计收益

- ## 2.样本选择

    - ### 2.1好处

        - 缩短模型的时间;具体问题不需要全部信息;去除噪声数据

    - ### 2.2数据噪声

        - 特征值缺失
        - 特征值超出值域范围
        - 误标注：二分类正样本标注为负样本
            - 集成过滤法(Ensembel filter,EF)
            - 交叉验证委员会过滤法(Cross-validated committees filter,CVCF)
            - 迭代分割过滤法(Iterative partitioning filter,IPF)
        - 清洗爬虫数据
        - 过滤无效曝光样本

    - ### 2.3采样

        - 好处
            - 克服高维特征以及大量数据导致的问题,有助于降低成本,缩短时间甚至提升效果
        - 采样方法：
            - 无放回的简单抽样：每条样本被采到的概率相等且都为1/N
            - 有放回的简单抽样：每条样本可能多次被选中
            - 平衡采样,考虑正负比例
                - 上采样
                    - SMOTE
                    - ADASYN
                - 下采样
            - 整体采样
                - 先将数据集T中的数据分组成G个互斥的簇,然后再从G个簇中简单随机采样s个簇作为样本集.
            - 分层采样
                - 数据集T划分成不同的层,然后在每层内部进行简单随机抽样,最后汇总成样本集合S.该方法也常用于不平衡分类问题中,对每个类别进行采样,能使每个类别在样本集S中的分布和数据集T中的分布更为一致.

    - ### 2.4原型选择

        - 基于实例的方法,在样本选择过程中不需要训练模型,而是选取相似度或距离度量标准来找到分类精度和数据量最佳的训练集,多数采用KNN算法
        - 根据从数据集T中选择样本集S的方法分类
            - 递增法
            - 递增法
            - 批量法
            - 混合法
            - 固定法
        - 根据选择样本的策略进行分类
            - 包装法：根据模型的目标函数,一般是模型预测结果来进行样本选择
            - 过滤法：样本的选择标准不基于模型
        - 根据选择的样本进行分类
            - condensation：保留决策边界处样本
            - Edition：删除边界的异常点,使得边界更平滑
            - Hybrid：尝试找到最小的样本集S,能够保持甚至提升测试集上的泛化精度

    - ### 2.5训练集选择

        - 构建预测模型来进行样本选择,比如决策树、ANN和SVM



- ## 3.交叉验证

    - ### 3.1留出法

        - 将数据集随机划分为互斥的训练集和测试集
        - 缺点：训练集和测试集的划分验证影响最终结果
        - 适用场景：数据有明显的时间序列因素,即线上数据的时间都在离线数据集之后
        - 多次留出法：考虑到单词留出法得到的结论往往不靠谱,我们进行多次留出法,每次随机划分,最终将多次得到的实验结论进行平均.

    - ### 3.2K折交叉验证

        - 比如分成5份,每次用其他的4份作为训练集,用该份进行验证;可以训练5次
        - K值太小实验稳定性偏低,K值太大有可能导致实验成本高,常用K为5,10
        - 留一法=N折交叉验证
            - K=N时,就是留一法(Leave-one-out,LLO)即每一条样本当测试集,其余数据作训练
            - 训练N个模型,每个模型都基本用到了全部的数据,得到的模型和全部数据D得到的模型更相似,并且不再受随机样本划分方式的影响,因为划分方式只有一种了.但是当数据量N很大时,计算成本非常高,当数据稀疏时,LOO很适用.
        - 分层K折交叉验证(stratified K-Fold)

    - ### 3.3自助法Boostraping

        - n个样本的样本集,重复有放回采样取n次,分成训练集和测试集
        - 缺点：自助法改变了初始数据集的分布,会引入估计偏差

    - ### 总结

        - 数据量足够时,一般采用留出法和交叉验证法;
        - 数据量较小时,并且难以有效区分训练集和测试集时,自助法很有用. 

- # 二.特征工程

    数据和特征决定了机器学习算法的上限,而模型和算法只是不断逼近这个上限而已.

- - ## 1.特征提取

        - ### 1.1探索性的数据分析（Exploratory Data Analysis）

            - 发现内部数据的规律、检测异常值.
            - 可视化技术：箱型图/直方图/多变量图/链图/帕累托图/散点图/茎叶图
            - 定量技术：均值/方差/分位数/峰度/偏度

        - ### 1.2数值特征

            - ##### 截断

                - 连续型的数值进行截断或者对长尾数据进行对数后截断(保留重要信息的前提下对特征进行截断,截断后的特征也可以看作是类别特征)

            - ##### 二值化

                - 如网站每天的访问量、餐厅的评论数、用户对一首歌的播放次数.大数据时代,计算可以非常快地增加,处理计算特征时,首先要考虑,保留为原始计数还是转换为二值变量来标识是否存在或者进行分桶操作.

            - ##### 分桶

                - 均匀分桶：如商品的评论次数、年龄
                - 分位数分桶：如果数值变量的取值存在很大间隔时,有些桶里没有数据,可以基于数据的分布进行分桶,例如利用聚类将特征分为多个类别.

            - ##### 缩放

                - 标准化缩放(Z缩放)
                - 最大最小值缩放
                - 最大绝对值缩放
                - 范数归一化：如使用L1范数、L2范数将数值向量的范数变为1
                - 平方根缩放、对数缩放
                    - 对数缩放适用于处理长尾分且取值为正数的数值变量
                    - 二者都是方差稳定的变换,幂变换的特例
                    - Box-Cox转换仅对取值为正数的特征起作用
                    - 对于有异常点的数据,可以使用更加健壮的缩放,与一般的标准化基于标准差进行缩放不同的是,健壮的缩放使用中位数而不是均值,基于分位数而不是方差.

            - ##### 缺失值处理

                - 补均值、中位数
                - 使用模型预测缺失值
                - 直接忽略,即将缺失作为一种信息进行编码喂给模型让其进行学习,现在有一些模型可以直接处理缺失值,如Xgboost模型可以处理缺失特征

            - ##### 特征交叉

                - 针对两个数值特征做加减乘除,可以通过特征选择方法(如统计检验或者模型的特征重要性)来选择有用的交叉组合.有些特征交叉组合,虽然没有直观的解释,但有可能对于模型效果有很大的提升
                - 自动进行特征交叉组合,如FM/FFM,特征交叉可以在线性模型中引入非线性模型,提升模型的表达能力.

            - ##### 非线性编码

                - 线性模型往往很难学习到数据中的非线性关系,除特征交叉之外,也可以通过非线性编码来提升线性模型的效果.例如使用多项式核,高斯核等
                - 将随机森林模型的叶节点进行编码喂给线性模型,这样线性模型的特征包含了复杂的非线性信息
                - 基因算法/局部线性嵌入/谱嵌入/t-SNE等

            - ##### 行统计量

                - 统计行向量中空值的个数、0的个数、正值或负值的个数
                - 均值、方差、最大值、最小值、偏度、峰度

        - ### 1.3类别特征

            - ##### 自然数编码

                - 含有大小关系的数据,比如学生考试成绩,分为高中低三档,1表示高,2表示中,3表示低

            - ##### 独热编码

                - 处理不具有大小关系的特征
                - 缺点：
                    - 会生成稀疏向量；
                    - KNN在高纬度下两点之间的距离很难进行有效的衡量,在LR模型中,参数的数量会随着维度的增高而增加,任意引起过拟合现象.

            - ##### 分层编码

                - 如邮政编码、身份证等,可以取不同维数进行分层,然后按层次进行自然数编码

            - ##### 散列编码

                - 取值特别多的类别特征,使用独热编码得到的特征矩阵非常稀疏,因此在进行独热编码之前可以先对类别进行散列编码,可以避免特征矩阵过于稀疏
                - 缺点：可能会导致特征取值冲突,这种冲突通常会削弱模型的效果,自然数编码和分层编码可以看作散列编码的特例.

            - ##### 计数编码

                - 将类别特征用其对应的计数来代替,这对线性和非线性模型都有效.
                - 缺点：对异常值比较敏感,特征取值有可能冲突.

            - ##### 计数排名编码

                - 利用计数的排名对类别特征进行编码,这对线性和非线性模型都有效
                - 缺点：对异常值不敏感,类别特征取值不会冲突

            - ##### 目标编码

                - 基于目标变量对类别特征进行编码,对于基数(类别变量所有可能不同取值的个数)很大的离散特征,例如IP地址、网站域名、城市名、家庭地址、街道、产品编号等,用自然数编码,简单模型任意欠拟合,而复杂模型任意过拟合;对于独热编码,得到的特征矩阵太稀疏.对于高基数类别变量,有效方式是基于目标变量对类别特征进行编码,即有监督的编码方式,适用于分类和回归问题.
                - 分类:采用交叉验证的方式,即将原本划分为5份,针对其中每一份数据,计算离散特征每个取值在另外4份数据中每个类别的比例.为了避免过拟合,也可以采用嵌套的交叉验证划分方法;回归问题同样采用交叉验证的方式计算目标变量均值对类别变量编码.
                - 目标编码对于基数较低的离散变量通常很有效,但对于基数特别高的离散数量,可能会有过拟合的风险,对于很多类别特征的取值样本个数太少,不具有统计意义.这种情况,通常采用贝叶斯方式,即对统计特征进行贝叶斯平滑,如拉普拉斯平滑或者先验概率和后验概率加权平均的方式.

            - ##### 类别特征之间交叉组合

                - 两个类别特征进行笛卡尔积操作：适用于两个(或多个)类别特征的基数较小的情况.
                - 基于统计的组合
                    - 如针对城市ID和商品ID两个类别特征,可以计算某个城市有多少不同的商品ID以及当前ID出现次数的分布,从而得到新的数值特征,或计算某个城市出现次数最多的商品ID,从而得到一个新的类别特征.
                    - 对于多个类别特征也可以,如针对年龄、性别、产品ID三个类别特征,可以计算某个年龄段不同性别的人购买过多少产品或者对当前产品ID购买次数的分布等.

            - ##### 类别特征和数值特征之间交叉组合

                - 通常是在某个类别中计算数值特征的一些统计量,如针对用户ID,统计过去一段时间内在网站上的浏览次数、购买次数、以及购买价格的统计量,如均值、中位数、标准差、最大值和最小值等;针对产品,统计用户对产品的评分、评价次数、购买次数、浏览次数等.
                - 例如统计产品在某个区域的销量、产品的价格、或者当前产品的价格跟产品所在区域内的平均价格的差价等.

        - ### 1.4时间特征

            - ##### 单个时间变量

                - 年、月、日、时、分、秒、星期几、年的第多少天、一天过了多少分钟、季度、是否闰年、是否季度初、是否季度末、是否周末、是否营业时间、是否节假日

            - ##### 两个时间变量之间

                - 计算产品上线到现在经过了多少时间、顾客上次借款距离现在的时间间隔、两个事件间隔是否包含节假日或其他特殊日期等.

            - ##### 时间序列相关的特征

                - 如股票价格、天气温度、降雨量、订单量等
                - 若问题是利用历史数据预测未来,则对于t时刻,可以将t-1、t-2和t-3时刻的值也作为特征使用,若问题可以考虑未来信息,则t+1、t+2、t+3时刻的值也可以作为特征使用【时间窗口宽度为1】

            - ##### 基于滑动窗口统计特征

                - 如计算前n个值的均值(回归问题),或者前n个值中每个类别的分布(分类问题),时间窗口的选取可以有多种方式,上面提到的滞后特征是滑动窗口统计的一种特例,另一种常用的窗口设置包含所有历史数据,称为扩展窗口统计.

        - ### 1.5空间特征

            - ##### 经纬度

                - 做散列处理,从而对空间区域进行分块,得到一个类别特征
                - 可以通过坐标拾取系统获得当前位置的行政区ID、街道ID、城市ID等类别特征,从而进行类别特征的处理方式进行特征预处理.

            - ##### 计算两个位置之间的距离

                - 如用户到超市或者电影院、餐厅的距离.距离的计算方式有很多种,例如可以计算欧氏距离、球面距离、曼哈顿距离,也可以是真实的街道距离.

        - ### 1.6文本特征

            - ##### 预处理

                - 将字符转化为小写、分词、去除无用字符、提取词根、词干提取、标点符号编码、文档特征、实体插入和提取、word2vec、文本相似性、去除停用词、去除稀有词、TF-IDF、LDA、LSA等

            - ##### 语料构建

                - 构建一个由文档或短语组成的矩阵,矩阵的每一行为文档,可以理解为对产品的描述,每一列为单词,通常,文档的个数与样本的个数一致

            - ##### 文本清洗

                - 如果数据通过网页抓取,首先提出文本中的HTML标记;
                - 停用词只用于语句的构建,但不包含任何真实的信息,因此需要剔除,
                - 为了避免文本中的大小写差异,整个文本通常转换为小写形式;
                - 统一编码;去除标点符号;去除数字;去除空格;还原为词根.
                - 但是在某些情况下,文本不一定需要进行清洗,这取决于具体的应用场景.例如考虑某编辑员对某物品的描述,如果我们关心的对象是物品,则需要去除噪声,保留关键信息,但如果我们关心的对象是编辑员,则噪声信息一定程度上反映了此编辑员的水平. 

            - ##### 分词

                - ###### 词性标注

                    - 名词、动词、形容词;词性标注可以帮助我们了解语言的内在结构

                - ###### 词形还原和词干提取

                    - 词性还原可以任何形式的语言还原为一般形式(能完整表达语义)
                    - 词干提取是抽取词的词干和词根形式(不一定能表达完整语义)

                - ###### 文本统计特征

                    - 不需要考虑词序信息,包括计算文本的长度、单词个数、数字个数、字母个数、大小写单词个数、大小写字母个数、标点符号个数、特殊字符个数等,数字占比、字母占比、特殊字符占比等,以及名词个数、动词个数等.

                - ###### N-Gram模型

                    - 将文本转换为连续序列,序列的每一项包含n个元素(可以是单词),这种想法是将一个或者两个甚至多个单词同时出现的信息喂给模型.3-Gram是常用的选择

            - ##### Skip-Gram模型

                - ###### 词集模型

                    - 将文本转换为实数或实向量;在词集模型中,向量中的每个分量的取值为0和1,代表单词是否在文档汇总出现,向量空间模型没有考虑词序信息

                - ###### 词袋模型

                    - 向量的每个分量的取值为单词在文档中的词频,为了避免向量维度太大,通常会过滤掉在文档集合中词频很小的单词.

                - ###### TF-IDF

                    - TF(Term Frequency,词频)
                    - IDF(Inverese Document Frequency逆文档频率),用于评估单词对于文件集或语料库中的其中一份文件的重要程度.
                    - 单词或短语的重要性随着它在文档中出现的次数成正比增加,同时随着它在语料库中出现的频率成反比下降.
                    - 可以基于文档的TF-IDF向量表示计算文档之间的相似度,但不能很好地表示特别长的文档,而且这种向量表示也没有考虑词序信息.基于TF-IDF和词袋模型得到的表示文本的向量往往维度非常大,因此实际应用中一般需要降维处理.

            - ##### 余弦相似度

                - 计算检索词q和文档d之间的相关性.如将检索词和文档都表示为向量,计算两个向量之间的余弦相似度.

            - ##### Jaccard相似度

                - 两个文档中相交的单词个数除以两个文档出现单词的总和.

            - ##### Levenshtein(编辑距离)

                - 指两个字符串由一个转成另外一个所需要的最少编辑操作(如插入、删除、替换)次数,它也是衡量两个字符串相似度的指标

            - ##### 隐性语义分析

                - 把高维的向量空间模型表示的文档映射到低维的潜在语义空间中,即采用将文档或词矩阵进行SVD分解,由于SVD分解本身是对文档特征进行排序,我们可以通过限制奇异值的个数对数据进行降噪和降维.一般而言,文档和文档或者文档和查询之间的相似性在简化的潜在语义空间的表达更为可靠.

            - ##### Word2Vec

                - 即将单词所在的空间映射到一个低维的向量空间中,这样每个单词对应一个向量,通过计算向量之间的余弦相似度就可以得到某个单词的同义词.

- - ## 2.特征选择

        - ### 2.1目的

            - 简化模型；提高性能,减少内存和计算开销;改善通用性、降低过拟合

        - ### 2.2前提

            - 训练数据中包含许多冗余或者无关的特征,移除这些特征并不会导致丢失信息

        - ### 2.3过程

            - 特征产生过程
            - 评价函数
                    - 衡量特征或特征子集的重要性或者好坏程度,因此需要量化特征变量和目标变量之间的联系以及特征之间的相互关系.为了避免过拟合,一般采用交叉验证的方式来评估特征的好坏
                - 停止准则
                    - 为了减少计算复杂度,可以设定一个阈值,当评价指标达到阈值后搜索停止
                - 验证过程
                    - 在验证集上验证选出来的特征子集的有效性

        - ### 2.4方法：

            - #### 2.4.1过滤法

                - ##### 单变量

                    - ###### 覆盖率

                        - 如果特征的覆盖率低,则可以剔除

                    - ###### 皮尔森相关系数

                        - 计算两个特征之间的线性相关性

                    - ###### Fisher得分

                        - 对于分类问题,Fisher得分越高,则特征在不同类别间的差异性越大,在同类别中的差异性越小,则特征越重要

                    - ###### 假设检验

                        - 假设特征变量和目标变量之间相互独立,将其作为H0假设,选择适当检验方法计算统计量,然后根据统计量确定P值做出统计推断;卡方统计值越大,特征相关性越高
                        - 特征变量=类别变量,目标变量=连续变量
                            - 方差分析(Analysis of Variance,ANOVA)
                        - 特征变量=连续变量,目标变量=连续变量
                            - 皮尔森卡方检验

                    - ###### 互信息(或KL散度/相对熵)

                        - 度量两个变量之间的相关性,互信息越大表明两个变量相关性越高;互信息为0,两个变量越独立.

                    由于单变量过滤方法只考虑了单特征变量与目标变量之间的相关性,因此选择的特征子集可能过于冗余.

                - ##### 多变量

                    - ###### 最小冗余最大相关性(mRMR)

                        - 考虑到了特征之间的冗余性,具体做法是对跟已选择特征的相关性较高的冗余特征进行惩罚.
                        - 不足：没有考虑到特征之间的组合可能与目标变量比较相关,某个特征一旦被选择了,在后续的步骤中不会被删除,

                    - ###### QPFS

                        - 通过二次规划求解,偏向于选择熵比较小的特征,这是因为特征自身的冗余性

                    - ###### SPEC

                        - 通过求解矩阵的主特征向量来求解,而且可以处理二阶的特征组合

                    - ###### 相关特征选择(CFS)

                        - 基于以下假设来评估特征集合的重要性:好的特征集合包含跟目标变量非常相关的特征,但这些特征之间彼此不相关;这里的相关性不一定是皮尔森相关系数或斯皮尔曼相关系数

                    - ###### CFS、MBF、FCBF

                - ##### 工具包

                    - ###### 数据量小

                        - sklearn里面的feature_selection模块

                    - ###### 数据量大

                        - Spark MLlib

                ​        由于过滤方法与具体的机器学习算法独立,因此过滤方法没有考虑选择的特征集合在具体机器学习算法上的效果.封装方法直接使用机器学习算法评估特征子集的效果,它可以检测出两个或多个特征之间的交互关系,而且选择的特征子集让模型的效果达到最优.

            - #### 2.4.2封装方法

                - ##### 完全搜索

                    - 思想：先用全部特征,然后去掉一个看看效果提升没

                    - ###### 穷举

                        - 广度优先搜索
                            - 时间复杂度太高,不实用
                        - 分支定界搜索
                            - 在穷举搜索的基础上加入了分支限界,若断定某些分支不可能搜索出比当前找到的最优解更优的解,则可能剪掉这些分支

                    - ###### 非穷举

                        - 定向搜索
                            - 首先选择N个得分最高的特征作为特征子集,将其加入一个限制最大长度的优先队列,每次从队列中取出得分最高的子集,然后穷举向该子集加入一个特征后产生的所有特征集,将这些特征集加入队列;
                        - 最优优先搜索
                            - 与定向搜索类似,唯一不同的是不限制优先队列的长度

                - ##### 启发式搜索

                    - 思想：先用少量特征,再慢慢加

                    - 方法：

                        - ###### 序列向前选择

                            - 特征子集从空集开始,每次只加入一个特征,贪心

                        - ###### 序列向后选择

                            - 特征子集从全集开始,每次删除一个特征

                        - ###### 双向搜索

                            - 同时使用序列向前选择和向后选择,当两者搜索到相同的特征子集时停止

                        - ###### 增L去R

                            - 若算法先删除R个特征,再增加L个特征

                        - ###### 序列浮动选择

                            - 每次选择添加和删除的特征个数不是固定的

                - ##### 随机搜索

                    - 思想：执行序列向前或者向后选择的时候,此算法随机选择特征子集
                    - 模拟退火、随机爬山、基因算法

                ​         过滤方法与机器学习算法相互独立,而且不需要交叉验证,计算效率比较高,但是过滤算法没有考虑机器学习算法的特点;封装方法使用预先定义的机器学习算法来评估特征子集的质量,需要很多次训练模型,计算效率很低;嵌入方法则将特征选择嵌入到模型的构建过程中,具有封装方法与机器学习算法相结合的优点,而且具有过滤方法计算效率高的优点,是实际应用中最常用的方法.

            - #### 2.4.3嵌入方法　

                - ##### Lasso回归

                    - 通过对回归系数添加L1惩罚项来防止过拟合,可以让特定的回归系数变为0,从而可以选择一个不包含那些系数的更简单的模型.lambta越大,回归系数越稀疏,lambta一般用交叉验证的方法来确定;任何广义线性模型如逻辑回归/FM/FFM/神经网络模型,都可以添加L1惩罚项

                - ##### 基于树模型

                    - 可以基于树模型中特征出现次数等指标对特征进行重要性排序;深度较浅的节点一般特征分类能力更强(可以将更多的样本区分开)
                    - 如决策树、随机森林、梯度提升树

                - ##### SVM

                    - 自带L2正则

                - ##### 工具包

                    - 一般机器学习包的线性模型都支持L1正则,如Spark MLlib和Sklearn
                    - sklearn中的随机森林;Xgboost包,支持根据不同指标(如增益或者分裂次数)对特征进行排序

- # 三.模型选择

    - ## 1.分类

        - 二分类
        - 多分类

    - ## 2.回归

          

- # 四.模型融合

    - ## 1.定义

        - 单模型即从数据训练而来的模型,也叫作基学习器、个体学习器、组件学习器
        - 如果单模型是同一种模型,叫作同质模型融合;否则称异质模型融合

    - ## 2.融合的意义

        - 统计的角度：平均假设,可以得到一个逼近f的优化假设
        - 计算的角度：同一份数据集,从不同的出发点进行计算
        - 表示的角度：模型融合使得假设空间扩大

    - ## 3.融合的理论

        - 模型的误差分歧：单模型的差异性越大,最后融合的效果就越好
        - 多样性独立：不一致度量、相关系数、Q统计、K统计、双次失败度量、KW差异、k度量、熵度量
        - 多样性增强：数据样本的扰动、属性的扰动、输出的扰动、算法参数的扰动

    - ## 4.融合的方法

        - 平均分
        - 投票法
        - Bagging
            - 采用重采样方式,各个模型之间相关性不高,可以降低方差
        - Stacking
            - 把第一组模型训练的结构作为数据再次进行训练

 
 
