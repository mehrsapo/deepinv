from deepinv.models import RidgeRegularizer
import torch


model=RidgeRegularizer()

"""
multiconv_dict=torch.load('../../deepinv/saved_model/saved_model_WCRR/multiconv.pt',map_location='cpu')
model.W.load_state_dict(multiconv_dict)
alpha_dict=torch.load('../../deepinv/saved_model/saved_model_WCRR/alpha.pt',map_location='cpu')
model.potential.alpha_spline.load_state_dict(alpha_dict)
mu_dict=torch.load('../../deepinv/saved_model/saved_model_WCRR/mu.pt',map_location='cpu')
model.potential.mu_spline.load_state_dict(mu_dict)
phi_plus_dict=torch.load('../../deepinv/saved_model/saved_model_WCRR/phi_plus.pt',map_location='cpu')
model.potential.phi_plus.load_state_dict(phi_plus_dict)
phi_minus_dict=torch.load('../../deepinv/saved_model/saved_model_WCRR/phi_minus.pt',map_location='cpu')
model.potential.phi_minus.load_state_dict(phi_minus_dict)

all_weights=model.state_dict()
torch.save(all_weights,'../../deepinv/saved_model/weights.pt')
"""

model.load_state_dict(torch.load('../../deepinv/saved_model/weights.pt'))
