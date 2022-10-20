# Swarm-intelligence-optimization-algorithm

[![PyPI](https://img.shields.io/pypi/v/swarm-algorithm)](https://pypi.org/project/swarm-algorithm/)
[![License](https://img.shields.io/pypi/l/swarm-algorithm.svg)](https://github.com/LucXiong/Swarm-intelligence-optimization-algorithm/blob/master/LICENSE)
![Python](https://img.shields.io/badge/python->=3.5-green.svg)
[![fork](https://img.shields.io/github/forks/LucXiong/Swarm-intelligence-optimization-algorithm?style=social)](https://github.com/LucXiong/Swarm-intelligence-optimization-algorithm/fork)
[![star](https://img.shields.io/github/stars/LucXiong/Swarm-intelligence-optimization-algorithm?style=social)](https://github.com/LucXiong/Swarm-intelligence-optimization-algorithm/star)
[![Downloads](https://pepy.tech/badge/swarm-algorithm)](https://pepy.tech/project/swarm-algorithm)
[![Discussions](https://img.shields.io/badge/discussions-green.svg)](https://github.com/LucXiong/Swarm-intelligence-optimization-algorithm/discussions)

种群算法复现，由于已由[大佬](https://github.com/guofei9987/scikit-opt)开发了种群算法的第三方库，包括退火算法(SA)、粒子群算法(PSO)、人工免疫算法(IA)、遗传算法(GA)、差分进化算法(DE)、人工鱼群算法(AFSA)、蚁群算法(ACA)，标<sup>*1</sup>的表示从中copy过来的，但是删除了其中部分带有约束的部分，所以如果需要带有约束的原始算法可以去[大佬](https://github.com/guofei9987/scikit-opt)开发的种群算法的第三方库。另一个已有的[第三方种群算法库](https://github.com/HaaLeo/swarmlib)包括布谷鸟搜索算法(CS)、萤火虫算法(FA2009)、灰狼算法(GWO)、鲸鱼算法(WOA)、人工蜂群算法(ABC)，标<sup>*2</sup>的表示从中copy过来的。如有冒犯或者侵权，可联系删除。此仓库新增了二者没有一些种群算法。
可安装此第三方库调用此仓库已复现种群算法，若有问题，可随时联系。

```python
pip install swarm-algorithm
```

遗传算法（DA）、差分进化算法（DE）、粒子群算法(PSO<sup>*1</sup>1995)、烟花算法(FA2010)、乌鸦搜索(CSA2016)、樽海鞘群算法(SSA2017)、缎蓝园丁鸟优化算法(SBO2017)、麻雀搜索算法(SSA2020)，狼群搜索算法(WPS2007, WPA2013)，正余弦优化算法(CSA2016)
## 1995 Particle Swarm Optimization(PSO)<sup>*1</sup>
Kennedy J, Eberhart R. Particle swarm optimization[C]// [Particle swarm optimization](https://ieeexplore.ieee.org/abstract/document/488968). Proceedings of ICNN'95 - International Conference on Neural Networks, 27 Nov.-1 Dec. 1995.4: 1942-8 vol.4.
```python
import swarm-algorithm
pso = swarm-algorithm.PSO(func, n_dim=20, pop=40, max_iter=150, lb=-1e5, ub=1e5, w=0.8, c1=0.5, c2=0.5)
# n_dim : 变量维数
# pop : 种群数量
# max_iter : 最大迭代次数
# lb : 变量下界，维数为n_dim的列表
# ub : 变量上届，维数为n_dim的列表
# w、c1、c2 : 粒子更新规则的相关参数
n_dim = 30
lb = [-100 for i in range(n_dim)]
ub = [100 for i in range(n_dim)]
demo_func = test_function.fu2
pop_size = 100
max_iter = 1000
pso = PSO(func=demo_func, n_dim=n_dim, pop=100, max_iter=1000, lb=lb, ub=ub, w=0.8, c1=0.5, c2=0.5)
best_x, bext_y = pso.run()
print(f'{demo_func(pso.gbest_x)}\t{pso.gbest_x}')
```
## 2007 Wolf Pack Search(WPS)
Yang C, Tu X, Chen J. [Algorithm of Marriage in Honey Bees Optimization Based on the Wolf Pack Search](https://www.computer.org/csdl/proceedings-article/ipc/2007/30060462/12OmNzC5T5U), Jeju Island, Korea, 2007: 462-7.
## 2009 Gravitational Search Algorithm（GSA）
Rashedi E., Nezamabadi-Pour H., [Saryazdi S. GSA: A Gravitational Search Algorithm[J]](https://www.sciencedirect.com/science/article/pii/S0020025509001200). Information Sciences, 2009, 179(13): 2232-48.
## 2010 Fireworks Algorithm(FA2010)
Tan Y, Zhu Y. [Fireworks Algorithm for Optimization](https://www.researchgate.net/publication/220704568_Fireworks_Algorithm_for_Optimization#:~:text=Inspired%20by%20observing%20fireworks%20explosion%2C%20a%20novel%20swarm,keeping%20diversity%20of%20sparks%20are%20also%20well%20designed.)[M]. //  Lecture Notes in Computer Science. City: Springer Berlin Heidelberg, 2010: 355-64[2021-12-08T08:42:21]. 
## 2013 Wolf Pack Algorithm(WPA)
吴虎胜, 张凤鸣, 吴庐山. [一种新的群体智能算法——狼群算法](https://oss.wanfangdata.com.cn/www/%E4%B8%80%E7%A7%8D%E6%96%B0%E7%9A%84%E7%BE%A4%E4%BD%93%E6%99%BA%E8%83%BD%E7%AE%97%E6%B3%95%E2%80%94%E2%80%94%E7%8B%BC%E7%BE%A4%E7%AE%97%E6%B3%95.ashx?isread=true&type=perio&resourceId=xtgcydzjs201311033&transaction=%7B%22id%22%3Anull%2C%22transferOutAccountsStatus%22%3Anull%2C%22transaction%22%3A%7B%22id%22%3A%221478253592753254400%22%2C%22status%22%3A1%2C%22createDateTime%22%3Anull%2C%22payDateTime%22%3A1641278095793%2C%22authToken%22%3A%22TGT-21025086-NNOedndoqocfHmrAEEpa2NPFM54hlRyq9Iydysp3Vdrm52ZAb0-my.wanfangdata.com.cn%22%2C%22user%22%3A%7B%22accountType%22%3A%22Group%22%2C%22key%22%3A%22shjtdxip%22%7D%2C%22transferIn%22%3A%7B%22accountType%22%3A%22Income%22%2C%22key%22%3A%22PeriodicalFulltext%22%7D%2C%22transferOut%22%3A%7B%22GTimeLimit.shjtdxip%22%3A3.0%7D%2C%22turnover%22%3A3.0%2C%22orderTurnover%22%3A3.0%2C%22productDetail%22%3A%22perio_xtgcydzjs201311033%22%2C%22productTitle%22%3Anull%2C%22userIP%22%3A%22202.120.11.15%22%2C%22organName%22%3Anull%2C%22memo%22%3Anull%2C%22orderUser%22%3A%22shjtdxip%22%2C%22orderChannel%22%3A%22pc%22%2C%22payTag%22%3A%22%22%2C%22webTransactionRequest%22%3Anull%2C%22signature%22%3A%22DDZm%2FVXekyWdH42BgPDeLSdnJXD5YPlUmPP6RP2%2B5eU5k97eueMNcfB2qDS7gmRqjIAbT8ocLpCg%5CnFfEPHohFBJ9J%2BFzviaDCPBw8d6hI01pf4vPVSg9Dd2I4TakD%2FYViqh584dU9xvUJbBOxU8%2BaFsyF%5CnDfCN60TgqVGcQgxpefQ%3D%22%2C%22delete%22%3Afalse%7D%2C%22isCache%22%3Afalse%7D)[J]. 系统工程与电子技术, 2013, 35(11): 2430-8.

[New swarm intelligence algorithm--wolf pack algorithm](https://www.researchgate.net/publication/264928582_New_swarm_intelligence_algorithm-wolf_pack_algorithm)
## 2016 Crow Search Algorithm(CSA)
Askarzadeh A. [A novel metaheuristic method for solving constrained engineering optimization problems: Crow search algorithm](https://www.sciencedirect.com/science/article/pii/S0045794916300475)[J]. Computers & Structures, 2016, 169: 1-12.
## 2017 Salp Swarm Algorithm(SSA)
Mirjalili S, Gandomi A H, Mirjalili S Z, et al. [Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems](https://www.sciencedirect.com/science/article/pii/S0965997816307736)[J]. Advances in Engineering Software, 2017, 114: 163-91.
## 2017 Sine Cosine Algorithm(SCA)
Mirjalili S. [SCA: A Sine Cosine Algorithm for solving optimization problems](https://www.sciencedirect.com/science/article/pii/S0950705115005043)[J]. Knowledge-Based Systems, 2016, 96: 120-33.
## 2017 Satin Bowerbird Optimizer(SBO)
Samareh Moosavi S H, Khatibi Bardsiri V. [Satin bowerbird optimizer: A new optimization algorithm to optimize ANFIS for software development effort estimation](https://www.sciencedirect.com/science/article/pii/S095219761730006)[J]. Engineering Applications of Artificial Intelligence, 2017, 60: 1-15.
## 2020 Sparrow Search Algorithm(SSA)
Xue J, Shen B. [A novel swarm intelligence optimization approach: sparrow search algorithm](https://www.tandfonline.com/doi/pdf/10.1080/21642583.2019.1708830)[J]. Systems Science & Control Engineering, 2020, 8(1): 22-34.

