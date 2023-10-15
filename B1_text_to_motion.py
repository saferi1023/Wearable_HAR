### PART 1 - Motion Generation ###


# Import necessary libraries
import sys
import clip
import torch
import numpy as np
import models.vqvae as vqvae
import models.t2m_trans as trans
import warnings
warnings.filterwarnings('ignore')


def text_2_motion(clip_text):
    # Load CLIP model
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False, download_root='./')
    clip.model.convert_weights(clip_model)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # Set command line args for model initialization
    sys.argv = ['GPT_eval_multi.py']
    import options.option_transformer as option_trans
    args = option_trans.get_args_parser()

    # Model hyperparameters
    args.dataname = 't2m'
    args.resume_pth = 'pretrained/VQVAE/net_last.pth'
    args.resume_trans = 'pretrained/VQTransformer_corruption05/net_best_fid.pth'
    args.down_t = 2
    args.depth = 3
    args.block_size = 51

    # Initialize VQVAE model
    net = vqvae.HumanVQVAE(args,
                        args.nb_code,
                        args.code_dim,
                        args.output_emb_width,
                        args.down_t,
                        args.stride_t,
                        args.width,
                        args.depth,
                        args.dilation_growth_rate)

    # Initialize transformer model
    trans_encoder = trans.Text2Motion_Transformer(num_vq=args.nb_code,
                                        embed_dim=1024,
                                        clip_dim=args.clip_dim,
                                        block_size=args.block_size,
                                        num_layers=9,
                                        n_head=16,
                                        drop_out_rate=args.drop_out_rate,
                                        fc_rate=args.ff_rate)

    # Load pretrained VQVAE weights
    print ('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    net.eval()
    net.cuda()

    # Load pretrained transformer weights
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
    trans_encoder.eval()
    trans_encoder.cuda()

    # Load pretrained normalization factors
    mean = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')).cuda()
    std = torch.from_numpy(np.load('./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')).cuda()

    # Tokenize input text
    text = clip.tokenize(clip_text, truncate=True).cuda()

    # Get CLIP embedding for text
    feat_clip_text = clip_model.encode_text(text).float()

    # Generate latent codes from text using transformer
    index_motion = trans_encoder.sample(feat_clip_text[0:1], False)

    # Decode latent codes to pose vectors
    pred_pose = net.forward_decoder(index_motion)

    # Postprocess pose vectors
    from utils.motion_process import recover_from_ric
    pred_xyz = recover_from_ric((pred_pose*std+mean).float(), 22)

    # Reshape to animation frames
    xyz = pred_xyz.reshape(1, -1, 22, 3)

    return xyz


#### Input the text and output_path
output_path = "/home/saferi/CODE/Wearable_HAR/output/positions_data/"
clip_text = ["a person is running"]

# TODO: Define a meaningful naming convention to name each action
file_name = "example.npy"

# Get the xyz file
positions_data = text_2_motion(clip_text)

# Save final data
np.save(output_path+file_name, positions_data.detach().cpu().numpy())

print("Successfully motion generated from text!!!")