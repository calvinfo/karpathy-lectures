import torch

torch.manual_seed(1337)

B, T, C = 4, 8, 2
x = torch.randn(B, T, C)  # batch, time, channels
print(x.shape)

# our trick here is that we want each token to know
# about the previous tokens in the context
# how should we do that? we can take the moving average

# we want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.zeros((B, T, C))  # create x-bag-of-words (not crossbow)
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1]
        xbow[b, t] = torch.mean(xprev, 0)

print(x[0], xbow[0])

# the trick is that we can be super efficient using matrix multiplication
torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, 1, keepdim=True)
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b

print('a=', a)
print('----')
print("b=", b)
print('----')
print("c=", c)

# now the same thing as before
weights = torch.tril(torch.ones(T, T))
weights = weights / weights.sum(1, keepdim=True)
print(weights.size())
xbow2 = weights @ x # (T, T) @ (B, T, C) --> (B, T, T) @ (B, T, C) --> (B, T, C)

print(torch.allclose(xbow, xbow2))

# version 3, with the softmax
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros(T, T)
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = torch.nn.functional.softmax(wei, dim=-1)
print(wei)

# version 4: self-attention!
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

tril = torch.tril(torch.ones(T,T))
wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = wei.softmax(-1)
out = wei @ x
print(out.shape)