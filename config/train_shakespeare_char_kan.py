# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char-kan'
eval_interval = 50 # keep frequent because we'll overfit
eval_iters = 10
log_interval = 5 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'shakespeare-char-kan'
wandb_run_name = 'nanogpt-ReLU_KAN'

dataset = 'shakespeare_char'
kan = "ReLU_KAN" #"efficient_KAN" or "ReLU_KAN" or  "MLP"

gradient_accumulation_steps = 8
batch_size = 2
block_size = 128 # context of up to 256 previous characters

n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.1

learning_rate = 1e-4 # with baby networks can afford to go a bit higher
max_iters = 10000
lr_decay_iters = 10000 # make equal to max_iters usually
min_lr = 1e-5 # learning_rate / 10 usually
beta2 = 0.95 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
device = 'cuda'  # run on cpu only
compile = False # do not torch compile the model
