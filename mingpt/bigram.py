from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)

def load(filename):
    # load tiny shakespeare
    with open(filename) as f:
        contents = f.read()
        return contents


def encode_data(data):
    vocab = sorted(list(set(data)))
    itos = {ix: ch for ix, ch in enumerate(vocab)}
    stoi = {ch: ix for ix, ch in enumerate(vocab)}
    def encode(s): return [stoi[ch] for ch in s]
    def decode(seq): return ''.join([itos[idx] for idx in seq])
    return encode, decode, len(vocab)


def split_data(data):
    split = floor(len(data) * 0.9)
    return data[:split], data[split:]


def get_batch(data, block_size=8, batch_size=4):
    indices = torch.randint(low=0, high=len(
        data)-block_size, size=(batch_size, 1))
    xs = torch.stack([data[index:index+block_size] for index in indices])
    ys = torch.stack([data[index+1:index+block_size+1] for index in indices])
    xs, ys = xs.to(device), ys.to(device)
    return xs, ys

@torch.no_grad()
def estimate_loss(model, data):
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        xs, ys = get_batch(data, block_size=block_size, batch_size=batch_size)
        logits, loss = model(xs, ys)
        losses[k] = loss.item()
    model.train()
    return losses.mean()
    

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # todo create an embedding table of vocab_size x vocab_size
        self.embeddings = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None) -> tuple[torch.Tensor, torch.Tensor]:
        # todo pull out the embedding table for idx
        logits = self.embeddings(idx)  # (B, T, C)

        # if we're not training, it's just generating text instead
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # get the logits for the current index
            logits, loss = self(idx)
            # focus only on the last step (the last index in the time dimension)
            logits = logits[:, -1, :]
            # apply the softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # sample one sample (one char) from the probabilities
            # that new sample is idx_next
            idx_next = torch.multinomial(probs, num_samples=1)
            # update our idx with the concat of the new token
            idx = torch.cat((idx, idx_next), dim=1)
            # iterate until we are done
        return idx

# MAIN execution


data = load('./tiny-shakespeare.txt')
encode, decode, vocab_size = encode_data(data)

encoded = torch.tensor(encode(data))
encoded_tr, encoded_val = split_data(encoded)

print(get_batch(encoded_tr))

m = BigramLanguageModel(vocab_size)
xb, yb = get_batch(encoded_tr)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

idx = torch.zeros((1,1), dtype=torch.torch.long)
print(decode(m.generate(idx, 100)[0].tolist()))

# TRAIN

optim = torch.optim.AdamW(m.parameters(), lr=learning_rate)
for iter in range(max_iters):

    # every so often, evaluate loss on train and val sets averaged against
    # multiple batches
    if iter % eval_interval == 0:
        loss_tr = estimate_loss(m, encoded_tr)
        loss_val = estimate_loss(m, encoded_val)
        print(f"step {iter}: train loss {loss_tr}, val loss {loss_val}")

    # get a batch of training data
    xb, yb = get_batch(encoded_tr, batch_size=batch_size, block_size=block_size)

    # evaluate the loss
    logits, loss = m(xb, yb)
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

# GENERATE
idx = torch.zeros((1,1), dtype=torch.torch.long)
print(decode(m.generate(idx, 100)[0].tolist()))
