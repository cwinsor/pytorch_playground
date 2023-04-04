import math
import torch
import pyro
import pyro.distributions as dist

def model(rain=None, sprinkler=None, grasswet=None):
    rain = pyro.sample('rain', dist.Bernoulli(0.2), obs=rain)
    sprinkler_probs = 0.4 * rain + 0.01 * (1 - rain)
    sprinkler = pyro.sample('sprinkler',dist.Bernoulli(sprinkler_probs), obs=sprinkler)
    grasswet_probs = 0. * (1 - sprinkler) * (1 - rain) + 0.8 * (1 - sprinkler) * rain \
        + 0.9 * sprinkler * (1 - rain) + 0.99 * sprinkler * rain
    pyro.sample('grasswet', dist.Bernoulli(grasswet_probs), obs=grasswet)

p_grasswet = pyro.poutine.block(model, hide=['rain', 'sprinkler'])
F, T = torch.tensor(0.), torch.tensor(1.)

print("p(grasswet=F|rain=F,sprinkler=F):",
      pyro.poutine.trace(p_grasswet).get_trace(F, F, F).log_prob_sum().exp().item())
print("p(grasswet=T|rain=F,sprinkler=F):",
      pyro.poutine.trace(p_grasswet).get_trace(F, F, T).log_prob_sum().exp().item())