import torch
import torch.nn as nn
from torch.nn import functional as F
import math
class GPTConfig:
    def __init__(self):
        self.block_size = 16      # 上下文长度：一次能看多少个字
        self.vocab_size = 1000    # 词表大小：模型认识多少个字
        self.n_layer = 6          # 层数：堆叠多少层 Decoder Block
        self.n_head = 4           # 头数：有多少个“分身”同时注意力
        self.n_embd = 512          # 嵌入维度：向量的宽度 (d_model)
        self.dropout = 0.1        # 防止死记硬背的随机丢弃率
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. 定义 Q, K, V 的线性投影层 (参数都在这里!)
        # 为了方便，我们把 Q,K,V 写在一个大矩阵里，然后切分
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 2. 输出投影层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # 3. 注册一个下三角掩码矩阵 (tril)，不作为参数更新
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()  # Batch, Time(Sequence Length), Channels(Embd)

        # 计算 Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 拆分为多头: (B, T, n_head, head_size) -> 转置为 (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # 核心注意力计算: (Q @ K^T) / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # ***关键步骤：应用 Mask***
        # 将掩码为 0 的位置填充为负无穷 (-inf)，这样 Softmax 后概率为 0
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        # 聚合结果: att @ V
        y = att @ v
        # 拼合多头
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        return self.c_proj(y)

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd), # 升维
            nn.GELU(),                                   # 激活函数
            nn.Linear(4 * config.n_embd, config.n_embd), # 降维回原样
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.sa = CausalSelfAttention(config) # 模块二
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffwd = FeedForward(config)        # 模块三

    def forward(self, x):
        # 残差连接：x + sublayer(norm(x))
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class SimpleLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1. 词嵌入表 (Token Embeddings)
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        # 2. 位置嵌入表 (Positional Embeddings) - 可学习的参数
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
        # 3. 堆叠 N 层 Transformer Blocks
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # 4. 最终的归一化层
        self.ln_f = nn.LayerNorm(config.n_embd)
        # 5. 语言模型头 (LM Head)：把向量映射回词表概率
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # 获取 Token 嵌入 + 位置嵌入
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)

        # 通过 Transformer 层
        x = self.blocks(x)
        x = self.ln_f(x)

        # 得到 logits (预测分数)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            # 计算损失函数 (Cross Entropy)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            # PyTorch 的交叉熵会自动进行 Softmax
            loss = F.cross_entropy(logits, targets)

        return logits, loss



# ============== 数据准备 ==============
def get_batch(data, config, batch_size=32):
    """
    从数据中随机采样一个批次
    data: 一维张量，包含所有训练数据的token ids
    返回: (input_batch, target_batch)
    """
    # 随机选择 batch_size 个起始位置
    ix = torch.randint(len(data) - config.block_size, (batch_size,))
    # 构造输入和目标
    x = torch.stack([data[i:i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + config.block_size + 1] for i in ix])
    return x.to(config.device), y.to(config.device)


# ============== 估计损失函数 ==============
@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, eval_iters=200):
    """
    在训练集和验证集上估计平均损失
    """
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, config)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ============== 生成文本 ==============
@torch.no_grad()
def generate(model, idx, max_new_tokens, config):
    """
    根据给定的上下文 idx 生成新的 tokens
    idx: (B, T) 当前的上下文序列
    max_new_tokens: 要生成多少个新 token
    """
    model.eval()
    for _ in range(max_new_tokens):
        # 截取最后 block_size 个 tokens (因为位置嵌入有限制)
        idx_cond = idx[:, -config.block_size:]
        # 前向传播获取 logits
        logits, _ = model(idx_cond)
        # 只关注最后一个时间步的预测
        logits = logits[:, -1, :]  # (B, vocab_size)
        # 应用 softmax 获取概率分布
        probs = F.softmax(logits, dim=-1)
        # 从分布中采样下一个 token
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
        # 拼接到序列后面
        idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
    model.train()
    return idx


# ============== 交互式对话 ==============
def interactive_chat(model, config, encode, decode):
    """
    交互式对话模式
    """
    model.eval()

    while True:
        user_input = input("\n你: ").strip()

        if user_input.lower() in ['quit', 'exit', '退出']:
            print("再见！")
            break

        if user_input.lower() == 'random':
            context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
            print("\n模型: ", end='')
        else:
            try:
                context_tokens = encode(user_input)
                context = torch.tensor([context_tokens], dtype=torch.long, device=config.device)
                print(f"\n模型: {user_input}", end='')
            except KeyError as e:
                print(f"错误：输入包含未知字符 {e}")
                continue

        generated = generate(model, context, max_new_tokens=200, config=config)
        generated_text = decode(generated[0].tolist())

        if user_input.lower() != 'random':
            generated_text = generated_text[len(user_input):]

        print(generated_text)


def load_model(model_path):
    """
    加载已保存的模型
    """
    checkpoint = torch.load(model_path, weights_only=False)  # 添加 weights_only=False
    config = checkpoint['config']

    model = SimpleLLM(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    model.eval()

    return model, config
# ============== 准备训练数据 ==============
def prepare_data(text, train_split=0.9):
    """
    将文本转换为 token ids
    返回: (train_data, val_data, encode, decode)
    """
    # 构建字符级词表
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    # 创建字符到索引的映射
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # 编码和解码函数
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    # 将整个文本编码
    data = torch.tensor(encode(text), dtype=torch.long)

    # 划分训练集和验证集
    n = int(train_split * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, encode, decode, vocab_size


# ============== 训练函数 ==============
def train(model, train_data, val_data, config,
          max_iters=5000,
          eval_interval=500,
          learning_rate=3e-4,
          batch_size=32):
    """
    训练模型
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print(f"开始训练，共 {max_iters} 步")
    print(f"设备: {config.device}")
    print("-" * 50)

    for iter in range(max_iters):
        # 每隔一段时间评估一次损失
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, config)
            print(f"步数 {iter:5d} | 训练损失: {losses['train']:.4f} | 验证损失: {losses['val']:.4f}")

        # 获取一个批次的数据
        xb, yb = get_batch(train_data, config, batch_size)

        # 前向传播
        logits, loss = model(xb, yb)

        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("-" * 50)
    print("训练完成！")
# 单独运行交互模式（跳过训练）
# ============== 主函数示例 ==============
if __name__ == "__main__":
    import os

    # 检查是否有已保存的模型
    model_exists = os.path.exists('simple_llm.pth')

    if model_exists:
        print("发现已保存的模型 simple_llm.pth")
        choice = input("选择模式：\n1. 直接使用已有模型交互\n2. 重新训练模型\n请输入 1 或 2: ").strip()

        if choice == '1':
            # 加载已保存的模型
            print("\n正在加载模型...")
            checkpoint = torch.load('simple_llm.pth')
            config = checkpoint['config']

            model = SimpleLLM(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(config.device)
            model.eval()

            # 加载文本数据以获取 encode/decode
            if os.path.exists('input.txt'):
                with open('input.txt', 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                print("错误：找不到 input.txt 文件")
                exit(1)

            _, _, encode, decode, _ = prepare_data(text)

            print("模型加载成功！")
            print("\n" + "=" * 50)
            print("进入交互模式！")
            print("=" * 50)
            print("输入提示词，模型会续写文本")
            print("输入 'quit' 或 'exit' 退出")
            print("输入 'random' 从随机位置生成")
            print("-" * 50)

            interactive_chat(model, config, encode, decode)
            exit(0)

    # 如果选择重新训练或没有模型，继续执行训练流程
    print("\n开始训练新模型...")

    # 1. 准备数据
    if os.path.exists('input.txt'):
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        print("使用本地 input.txt 文件")
    else:
        print("未找到 input.txt，尝试下载莎士比亚文本...")
        try:
            import urllib.request

            url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            with urllib.request.urlopen(url) as response:
                text = response.read().decode('utf-8')
            # 保存到本地
            with open('input.txt', 'w', encoding='utf-8') as f:
                f.write(text)
            print("莎士比亚文本下载成功！")
        except Exception as e:
            print(f"下载失败: {e}")
            print("使用内置示例文本...")
            text = """To be, or not to be, that is the question.
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. """ * 100

    train_data, val_data, encode, decode, vocab_size = prepare_data(text)

    # 2. 创建配置和模型
    config = GPTConfig()
    config.vocab_size = vocab_size

    model = SimpleLLM(config)
    model = model.to(config.device)

    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 3. 训练模型
    train(model, train_data, val_data, config,
          max_iters=5000,
          eval_interval=500,
          learning_rate=3e-4,
          batch_size=64)

    # 4. 生成文本示例
    print("\n" + "=" * 50)
    print("生成文本示例:")
    print("=" * 50)

    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    generated_ids = generate(model, context, max_new_tokens=500, config=config)[0].tolist()
    generated_text = decode(generated_ids)
    print(generated_text)

    # 5. 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab_size': vocab_size,
    }, 'simple_llm.pth')
    print("\n模型已保存到 simple_llm.pth")

    # 6. 进入交互模式
    print("\n" + "=" * 50)
    print("进入交互模式！")
    print("=" * 50)
    print("输入提示词，模型会续写文本")
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'random' 从随机位置生成")
    print("-" * 50)

    interactive_chat(model, config, encode, decode)