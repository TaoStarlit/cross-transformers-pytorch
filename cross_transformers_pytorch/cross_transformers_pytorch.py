import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange

class CrossTransformer(nn.Module):
    def __init__(
        self,
        dim = 512,
        dim_key = 128,
        dim_value = 128
    ):
        super().__init__()
        self.scale = dim_key ** -0.5
        self.to_qk = nn.Conv2d(dim, dim_key, 1, bias = False)
        self.to_v = nn.Conv2d(dim, dim_value, 1, bias = False)

    def forward(self, model, img_query, img_supports):
        """
        dimensions names:
        
        b - batch
        k - num classes
        n - num images in a support class
        c - channels
        h, i - height
        w, j - width
        """

        b, k, *_ = img_supports.shape

        query_repr = model(img_query)
        *_, h, w = query_repr.shape

        img_supports = rearrange(img_supports, 'b k n c h w -> (b k n) c h w', b = b)
        supports_repr = model(img_supports)

        query_q, query_v = self.to_qk(query_repr), self.to_v(query_repr)

        supports_k, supports_v = self.to_qk(supports_repr), self.to_v(supports_repr)  # map both query and support sets' representation to qk and v by the identical conv2d(512,128,1, no bias)
        supports_k, supports_v = map(lambda t: rearrange(t, '(b k n) c h w -> b k n c h w', b = b, k = k), (supports_k, supports_v)) # for support set b: b k n 
        # 第一个 einsum 是找到所有 每个类别的support set Key里面，和 query Q内积最大的例子和位置, softmax就形成了这样的attention
        sim = einsum('b c h w, b k n c i j -> b k h w n i j', query_q, supports_k) * self.scale # 乘积的和， 4 5 -- 7 少了两个就是 c, c通道进行内积。 除了两个b合并一个，其他都保留，可以理解为b就等于1
        sim = rearrange(sim, 'b k h w n i j -> b k h w (n i j)') # b k h w  (n i j)  n i j 是support set的长和宽

        attn = sim.softmax(dim = -1) # 对support set的   N H W进行softmax， 为什么不对K呢？　　所以每个类里，找打最接近的例子最接近的位置，　这个就是attention？ 对整个范围做内积和softmax
        attn = rearrange(attn, 'b k h w (n i j) -> b k h w n i j', i = h, j = w) # 然后再还原 为什么 i=h j=w要表明呢？  因为你从一个向量重排为一个矩阵，有多种方式
        # 第二个 einsum 是 key的attention, 和 support set的 value,   n i j都内积掉， 剩下query的 位置， supportV的通道类别。   -- query某个位置：support类别最大的地方hw, 通道c
        out = einsum('b k h w n i j, b k n c i j -> b k c h w', attn, supports_v) # n i j , c i j 都不见了，剩下 attention的b k h w 和 value c,  两边的 n i j 都各自内积? 
        
        # bkchw query v也是bchw 没有K， 做一下欧式距离得到  c h w 加到一一起， 剩下 B K，就是最接近某个类的距离
        out = rearrange(out, 'b k c h w -> b k (c h w)')
        query_v = rearrange(query_v, 'b c h w -> b () (c h w)')

        euclidean_dist = ((query_v - out) ** 2).sum(dim = -1) / (h * w)
        return -euclidean_dist
