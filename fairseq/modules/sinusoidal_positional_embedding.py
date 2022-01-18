# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from decimal import Decimal
import math
from typing import Any, List, Optional

import torch
from torch import Tensor, nn
import torch.onnx.operators

from fairseq import utils


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        )
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.max_positions = int(1e5)

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    @staticmethod
    def get_embedding(
        num_embeddings_i: int, embedding_dim_i: int, padding_idx_i: Optional[int] = None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        def log(x: str):
            import logging
            logging.getLogger(__name__).debug("[SinusoidalPositionalEmbedding|get_embedding]: " + x)
        
        log("Building sinusoidal embeddings with args: " + str([num_embeddings_i, embedding_dim_i, padding_idx_i]))

        num_embeddings = Decimal(num_embeddings_i)
        embedding_dim = Decimal(embedding_dim_i)
        padding_idx = Decimal(padding_idx_i) if padding_idx_i is not None else None
                                                                                                                           
        half_dim = embedding_dim // 2
        emb = Decimal(10000).ln() / (half_dim - 1)
                                                                            
        #Prevent floating point nonsense by setting to float64
        emb = [(-emb * Decimal(x)).exp() for x in range(int(half_dim))]
        emb = torch.tensor([[x * y for y in emb] for x in range(int(num_embeddings))], dtype=torch.float64)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        emb = emb.view(
            int(num_embeddings), -1
        )
                                                                                                                           
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(int(num_embeddings), 1)], dim=1)
                                                                                                                           
        if padding_idx is not None:
            emb[int(padding_idx), :] = 0
        
        return emb

    def forward(
        self,
        input,
        incremental_state: Optional[Any] = None,
        timestep: Optional[Tensor] = None,
        positions: Optional[Any] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bspair = torch.onnx.operators.shape_as_tensor(input)
        bsz, seq_len = bspair[0], bspair[1]
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            if self.onnx_trace:
                return (
                    self.weights.index_select(index=self.padding_idx + pos, dim=0)
                    .unsqueeze(1)
                    .repeat(bsz, 1, 1)
                )
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = utils.make_positions(
            input, self.padding_idx, onnx_trace=self.onnx_trace
        )
        if self.onnx_trace:
            flat_embeddings = self.weights.detach().index_select(0, positions.view(-1))
            embedding_shape = torch.cat(
                (bsz.view(1), seq_len.view(1), torch.tensor([-1], dtype=torch.long))
            )
            embeddings = torch.onnx.operators.reshape_from_tensor_shape(
                flat_embeddings, embedding_shape
            )
            return embeddings
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )
