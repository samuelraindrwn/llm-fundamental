import random
import math

random.seed(42)

text = "aku suka koding ai." 

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

print(f"Dataset: '{text}'")
print(f"Vocab: {chars}")

data = [stoi[c] for c in text]
block_size = 3
x = []
y = []

for i in range(len(data) - block_size):
    x.append(data[i : i + block_size])
    y.append(data[i + block_size])

n_embd = 4      
n_hidden = 48
epochs = 5000
lr = 0.05

input_dim = n_embd * block_size

scale_w1 = (2/input_dim)**0.5
scale_w2 = (2/n_hidden)**0.5

embedding_table = [[random.gauss(0, 0.1) for _ in range(n_embd)] for _ in range(vocab_size)]
position_embedding_table = [[random.gauss(0, 0.1) for _ in range(n_embd)] for _ in range(block_size)]

W1 = [[random.gauss(0, scale_w1) for _ in range(n_hidden)] for _ in range(input_dim)]
W2 = [[random.gauss(0, scale_w2) for _ in range(vocab_size)] for _ in range(n_hidden)]

b1 = [0.0] * n_hidden
b2 = [0.0] * vocab_size

def dot(v1, v2):
    return sum(x*y for x, y in zip(v1, v2))

def outer(v1, v2):
    return [[x * y for y in v2] for x in v1]

def relu(v):
    return [max(0, x) for x in v]

def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    s = sum(exps)
    return [e/s for e in exps]

x_train = []
for sequence in x:
    seq_vecs = []
    for t, idx in enumerate(sequence):
        v = [embedding_table[idx][j] + position_embedding_table[t][j] for j in range(n_embd)]
        seq_vecs.extend(v) 
    x_train.append(seq_vecs)

print("\n--- MULAI TRAINING ---")

for step in range(epochs):
    loss_sum = 0
    
    for i in range(len(x_train)):
        input_vec = x_train[i]
        target_idx = y[i]
        
        cols_W1 = list(zip(*W1))
        h_pre = [dot(input_vec, col) + b1[j] for j, col in enumerate(cols_W1)]
        h = relu(h_pre)
        
        cols_W2 = list(zip(*W2))
        logits = [dot(h, col) + b2[j] for j, col in enumerate(cols_W2)]
        
        probs = softmax(logits)
        loss = -math.log(probs[target_idx] + 1e-9)
        loss_sum += loss
        
        dlogits = list(probs)
        dlogits[target_idx] -= 1
        
        dW2 = outer(h, dlogits)
        db2 = dlogits
        
        dh = [dot(dlogits, W2[j]) for j in range(n_hidden)]
        dh_pre = [dh[j] if h_pre[j] > 0 else 0 for j in range(n_hidden)]
        
        dW1 = outer(input_vec, dh_pre)
        db1 = dh_pre
        
        for r in range(n_hidden):
            for c in range(vocab_size):
                W2[r][c] -= lr * dW2[r][c]
        
        for c in range(vocab_size):
            b2[c] -= lr * db2[c]
        
        for r in range(input_dim):
            for c in range(n_hidden):
                W1[r][c] -= lr * dW1[r][c]
        
        for c in range(n_hidden):
            b1[c] -= lr * db1[c]

    if step % 1000 == 0:
        print(f"Epoch {step}: Loss = {loss_sum/len(x_train):.4f}")

print("Training Selesai!")

print("\n--- HASIL GENERATE TEKS ---")
start_word = "aku"
curr_idx = [stoi[c] for c in start_word]

print(f"Start: '{start_word}'", end="", flush=True)

for _ in range(50):
    input_seq = curr_idx[-block_size:]
    
    flat_vec = []
    for t, idx in enumerate(input_seq):
        v = [embedding_table[idx][j] + position_embedding_table[t][j] for j in range(n_embd)]
        flat_vec.extend(v)
        
    cols_W1 = list(zip(*W1))
    h_pre = [dot(flat_vec, col) + b1[j] for j, col in enumerate(cols_W1)]
    h = relu(h_pre)
    
    cols_W2 = list(zip(*W2))
    logits = [dot(h, col) + b2[j] for j, col in enumerate(cols_W2)]
    
    probs = softmax(logits)
    best_idx = probs.index(max(probs))
    
    char = itos[best_idx]
    
    if char == '.':
        break
        
    print(char, end="", flush=True)
    curr_idx.append(best_idx)

print("\n\n--- SELESAI ---")