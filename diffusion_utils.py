import torch
# def predict_start_from_noise(scheduler, timestep, sample, model_output):
#     # 2. compute alphas, betas
#     import pdb 
#     pdb.set_trace()
#     alpha_prod_t = scheduler.alphas_cumprod[timestep]
#     beta_prod_t = 1 - alpha_prod_t
#     pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)

#     return pred_original_sample


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def predict_start_from_noise(scheduler, timestep, sample, model_output):
    sqrt_recip_alphas_cumprod = torch.sqrt(1. / scheduler.alphas_cumprod).to(model_output.device)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / scheduler.alphas_cumprod - 1).to(model_output.device)

    return (
            extract_into_tensor(sqrt_recip_alphas_cumprod, timestep, sample.shape) * sample -
            extract_into_tensor(sqrt_recipm1_alphas_cumprod, timestep, sample.shape) * model_output
    )
# self.sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod))
# sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)