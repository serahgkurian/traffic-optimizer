 Traffic congestion at urban intersections leads to excessive delays, fuel consumption, and
 emissions under static timing plans. We introduce a continuous-action policy-gradient
 controller based on the REINFORCE algorithm, augmented with a learned value baseline
 and entropy regularization to stabilize learning. The agent directly learns Gaussian
parameterized green-light durations from a rich 60-dimensional state embedding that
 captures per-lane queue lengths, vehicle speeds, waiting times, flow rates, and cyclic
 phase encodings. To address multiple objectives—throughput maximization, wait-time
 minimization, efficiency, and fairness—we design an adaptive, multi-term reward whose
 weights respond dynamically to recent traffic trends. We validate our approach on a
 realistic four-way SUMO intersection with heterogeneous traffic and benchmark against
 a fixed-time controller. Results demonstrate 12.6% reduction in average waiting time and
 1.2% increase in throughput, showcasing the potential of variance-reduced, continuous
control RL for adaptive urban signal timing.
