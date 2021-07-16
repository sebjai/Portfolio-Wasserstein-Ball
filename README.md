This is the code for implementing the results in the paper:

Title: **Portfolio Optimisation within a Wasserstein Ball**
by    
Silvana Pesenti and Sebastian Jaimungal    
Department of Statistical Sciences, University of Toronto

https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3744994

Abstract: *We consider the problem of active portfolio management where a loss-averse and/or gain-seeking investor aims to outperform a benchmark strategy's risk profile while not deviating too much from it. Specifically, an investor considers alternative strategies that co-move with the benchmark and whose terminal wealth lies within a Wasserstein ball surrounding it. The investor then chooses the alternative strategy that minimises their personal risk preferences, modelled in terms of a distortion risk measure. In a general market model, we prove that an optimal dynamic strategy exists and is unique, and provide its characterisation through the notion of isotonic projections. Finally, we illustrate how investors with different risk preferences invest and improve upon the benchmark using the Tail Value-at-Risk, inverse S-shaped distortion risk measures, and lower- and upper-tail risk measures as examples. We find that investors' optimal terminal wealth distribution has larger probability masses in regions that reduce their risk measure relative to the benchmark while preserving some aspects of the benchmark.*

The code generates optimal portfolios allocations with CoIn, Gumbel, and no copula constraints for the stochastic interest rate - constant elasticity of variance model.

Please click on the WassersteinBall.ipynb and, if you wish, click on the colab link at the top to run the code live.
