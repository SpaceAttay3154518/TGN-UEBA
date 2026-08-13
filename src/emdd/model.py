"""Causal TGN detector with exact temporal snapshots and baseline hooks.

The implementation follows the pre-update scoring order used by the PyG TGN
example and KAIROS: score an observed interaction from past state, then update
memory and the temporal neighborhood with the ground-truth event.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import TransformerConv
from torch_geometric.nn.models.tgn import (
    IdentityMessage,
    LastAggregator,
    LastNeighborLoader,
    TGNMemory,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class TemporalAttentionEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        msg_dim: int,
        time_encoder: nn.Module,
        heads: int = 8,
        memory_decay_beta: float = 0.0,
        layers: int = 2,
        kairos_attention_layout: bool = False,
    ) -> None:
        super().__init__()
        self.time_encoder = time_encoder
        self.memory_decay_beta = float(memory_decay_beta)
        self.layers = int(layers)
        self.kairos_attention_layout = bool(kairos_attention_layout)
        if self.layers not in {1, 2}:
            raise ValueError("Temporal attention supports one or two layers")
        edge_dim = msg_dim + time_encoder.out_channels
        if not self.kairos_attention_layout and out_channels % heads != 0:
            raise ValueError("embedding_dim must be divisible by attention_heads")
        first_head_channels = (
            out_channels if self.kairos_attention_layout else out_channels // heads
        )
        first_width = first_head_channels * heads
        self.conv1 = TransformerConv(
            in_channels,
            first_head_channels,
            heads=heads,
            dropout=0.0,
            edge_dim=edge_dim,
        )
        self.conv2 = (
            TransformerConv(
                first_width,
                out_channels,
                heads=1,
                concat=False,
                dropout=0.0,
                edge_dim=edge_dim,
            )
            if self.layers == 2
            else None
        )

    def forward(
        self,
        x: Tensor,
        last_update: Tensor,
        edge_index: Tensor,
        edge_time: Tensor,
        edge_msg: Tensor,
        query_time: Tensor | None = None,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, tuple[Tensor, Tensor]]:
        if self.memory_decay_beta > 0 and query_time is not None:
            elapsed_days = (query_time.max() - last_update).clamp_min(0).to(x.dtype) / 86_400.0
            x = x * torch.exp(-self.memory_decay_beta * elapsed_days).unsqueeze(-1)
        if edge_index.numel() == 0:
            edge_attr = x.new_empty(
                (0, edge_msg.size(-1) + self.time_encoder.out_channels)
            )
        else:
            relative_time = last_update[edge_index[0]] - edge_time
            time_features = self.time_encoder(relative_time.to(x.dtype))
            edge_attr = torch.cat([time_features, edge_msg], dim=-1)
        first = self.conv1(x, edge_index, edge_attr)
        if self.conv2 is None:
            return first
        hidden = F.relu(first)
        if return_attention:
            assert self.conv2 is not None
            return self.conv2(
                hidden, edge_index, edge_attr, return_attention_weights=True
            )
        assert self.conv2 is not None
        return F.relu(self.conv2(hidden, edge_index, edge_attr))


class GuardedLastAggregator(nn.Module):
    """Last-message aggregation with an optional memory-similarity gate.

    The generated TGN message starts with source and destination memories.
    A positive temperature activates a smooth GNNGuard-style attenuation.
    Temperature zero is exactly the ordinary LastAggregator.
    """

    def __init__(self, memory_dim: int, temperature: float = 0.0) -> None:
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.temperature = float(temperature)
        self.last = LastAggregator()

    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int) -> Tensor:
        selected = self.last(msg, index, t, dim_size)
        if self.temperature <= 0 or selected.numel() == 0:
            return selected
        src = selected[:, : self.memory_dim]
        dst = selected[:, self.memory_dim : 2 * self.memory_dim]
        cosine = F.cosine_similarity(src, dst, dim=-1, eps=1e-8)
        gate = torch.sigmoid(cosine / self.temperature).unsqueeze(-1)
        populated = selected.abs().sum(dim=-1, keepdim=True) > 0
        return torch.where(populated, selected * gate, selected)


class LinkScorer(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.source = nn.Linear(channels, channels)
        self.destination = nn.Linear(channels, channels)
        self.output = nn.Linear(channels, 1)

    def forward(self, source: Tensor, destination: Tensor) -> Tensor:
        hidden = F.relu(self.source(source) + self.destination(destination))
        return self.output(hidden).flatten()


class EdgeTypeDecoder(nn.Module):
    """KAIROS-style endpoint decoder for relation reconstruction."""

    def __init__(self, channels: int, edge_types: int, dropout: float) -> None:
        super().__init__()
        self.source = nn.Linear(channels, channels * 2)
        self.destination = nn.Linear(channels, channels * 2)
        self.network = nn.Sequential(
            nn.Linear(channels * 4, channels * 8),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(channels * 8, channels * 2),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(channels * 2, max(channels // 2, edge_types)),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(max(channels // 2, edge_types), edge_types),
        )

    def forward(self, source: Tensor, destination: Tensor) -> Tensor:
        hidden = torch.cat(
            [self.source(source), self.destination(destination)], dim=-1
        )
        return self.network(hidden)


@dataclass
class Batch:
    src: Tensor
    dst: Tensor
    t: Tensor
    msg: Tensor
    neg_dst: Tensor | None = None
    edge_type: Tensor | None = None

    @property
    def size(self) -> int:
        return int(self.src.numel())

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            src=self.src.to(device),
            dst=self.dst.to(device),
            t=self.t.to(device),
            msg=self.msg.to(device),
            neg_dst=None if self.neg_dst is None else self.neg_dst.to(device),
            edge_type=None if self.edge_type is None else self.edge_type.to(device),
        )


class TGNDetector(nn.Module):
    """Memory-based continuous-time link predictor with recent-neighbor attention."""

    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        embedding_dim: int,
        neighbor_size: int,
        device: torch.device,
        num_edge_types: int = 0,
        attention_heads: int = 2,
        edge_type_loss_weight: float = 0.0,
        link_loss_weight: float = 1.0,
        decoder_dropout: float = 0.0,
        guard_temperature: float = 0.0,
        memory_decay_beta: float = 0.0,
        embedding_layers: int = 2,
        kairos_attention_layout: bool = False,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = memory_dim
        self.time_dim = time_dim
        self.embedding_dim = embedding_dim
        self.neighbor_size = neighbor_size
        self.num_edge_types = int(num_edge_types)
        self.attention_heads = int(attention_heads)
        self.edge_type_loss_weight = float(edge_type_loss_weight)
        self.link_loss_weight = float(link_loss_weight)
        self.decoder_dropout = float(decoder_dropout)
        self.device_ref = device
        self.guard_temperature = float(guard_temperature)
        self.memory_decay_beta = float(memory_decay_beta)
        self.embedding_layers = int(embedding_layers)
        self.kairos_attention_layout = bool(kairos_attention_layout)

        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(raw_msg_dim, memory_dim, time_dim),
            aggregator_module=GuardedLastAggregator(memory_dim, guard_temperature),
        )
        self.embedding = TemporalAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=raw_msg_dim,
            time_encoder=self.memory.time_enc,
            heads=attention_heads,
            memory_decay_beta=memory_decay_beta,
            layers=embedding_layers,
            kairos_attention_layout=kairos_attention_layout,
        )
        self.scorer = LinkScorer(embedding_dim)
        self.edge_decoder = (
            EdgeTypeDecoder(embedding_dim, num_edge_types, decoder_dropout)
            if num_edge_types > 0
            else None
        )
        self.neighbor_loader = LastNeighborLoader(
            num_nodes=num_nodes,
            size=neighbor_size,
            device=device,
        )
        self.register_buffer(
            "association", torch.empty(num_nodes, dtype=torch.long, device=device)
        )
        self._history_t = torch.empty(0, dtype=torch.long, device=device)
        self._history_msg = torch.empty(
            (0, raw_msg_dim), dtype=torch.float, device=device
        )
        self._history_length = 0
        self.to(device)

    def reset_temporal_state(self) -> None:
        self.memory.reset_state()
        self.neighbor_loader.reset_state()
        self._set_history(
            torch.empty(0, dtype=torch.long, device=self.device_ref),
            torch.empty(
                (0, self.raw_msg_dim), dtype=torch.float, device=self.device_ref
            ),
        )

    @property
    def history_t(self) -> Tensor:
        return self._history_t[: self._history_length]

    @property
    def history_msg(self) -> Tensor:
        return self._history_msg[: self._history_length]

    def _set_history(self, timestamp: Tensor, message: Tensor) -> None:
        length = int(timestamp.numel())
        capacity = max(length, 1024)
        self._history_t = torch.empty(
            capacity, dtype=torch.long, device=self.device_ref
        )
        self._history_msg = torch.empty(
            (capacity, self.raw_msg_dim), dtype=torch.float, device=self.device_ref
        )
        if length:
            self._history_t[:length].copy_(timestamp.to(self.device_ref))
            self._history_msg[:length].copy_(message.to(self.device_ref))
        self._history_length = length

    def _ensure_history_capacity(self, required: int) -> None:
        if required <= self._history_t.size(0):
            return
        capacity = max(required, max(self._history_t.size(0) * 2, 1024))
        timestamp = torch.empty(
            capacity, dtype=torch.long, device=self.device_ref
        )
        message = torch.empty(
            (capacity, self.raw_msg_dim), dtype=torch.float, device=self.device_ref
        )
        if self._history_length:
            timestamp[: self._history_length].copy_(self.history_t)
            message[: self._history_length].copy_(self.history_msg)
        self._history_t = timestamp
        self._history_msg = message

    def compact_history(self) -> None:
        """Retain only edge messages still referenced by sampled neighbors."""
        edge_ids = self.neighbor_loader.e_id
        mask = edge_ids >= 0
        if not bool(mask.any()):
            self._set_history(
                torch.empty(0, dtype=torch.long, device=self.device_ref),
                torch.empty(
                    (0, self.raw_msg_dim), dtype=torch.float, device=self.device_ref
                ),
            )
            self.neighbor_loader.cur_e_id = 0
            return
        retained = edge_ids[mask].unique(sorted=True)
        timestamp = self.history_t[retained].detach().clone()
        message = self.history_msg[retained].detach().clone()
        edge_ids[mask] = torch.searchsorted(retained, edge_ids[mask])
        self._set_history(timestamp, message)
        self.neighbor_loader.cur_e_id = int(len(retained))

    def fork(self) -> "TGNDetector":
        """Clone parameters and every component of online temporal state.

        ``copy.deepcopy`` is not supported for non-leaf tensors in current
        PyTorch. An explicit clone also makes the experimental checkpoint
        boundary auditable.
        """
        clone = TGNDetector(
            num_nodes=self.num_nodes,
            raw_msg_dim=self.raw_msg_dim,
            memory_dim=self.memory_dim,
            time_dim=self.time_dim,
            embedding_dim=self.embedding_dim,
            neighbor_size=self.neighbor_size,
            device=self.device_ref,
            num_edge_types=self.num_edge_types,
            attention_heads=self.attention_heads,
            edge_type_loss_weight=self.edge_type_loss_weight,
            link_loss_weight=self.link_loss_weight,
            decoder_dropout=self.decoder_dropout,
            guard_temperature=self.guard_temperature,
            memory_decay_beta=self.memory_decay_beta,
            embedding_layers=self.embedding_layers,
            kairos_attention_layout=self.kairos_attention_layout,
        )
        # Switch mode before loading buffers: TGNMemory performs a flush on an
        # eval transition, which must not alter the source checkpoint.
        clone.train(self.training)
        clone.load_state_dict(self.state_dict())

        def clone_store(store: dict) -> dict:
            return {
                int(node): tuple(tensor.detach().clone() for tensor in values)
                for node, values in store.items()
            }

        clone.memory.msg_s_store = clone_store(self.memory.msg_s_store)
        clone.memory.msg_d_store = clone_store(self.memory.msg_d_store)
        clone.neighbor_loader.neighbors.copy_(self.neighbor_loader.neighbors)
        clone.neighbor_loader.e_id.copy_(self.neighbor_loader.e_id)
        clone.neighbor_loader._assoc.copy_(self.neighbor_loader._assoc)
        clone.neighbor_loader.cur_e_id = self.neighbor_loader.cur_e_id
        clone._set_history(
            self.history_t.detach().clone(), self.history_msg.detach().clone()
        )
        return clone

    def _query_embeddings(
        self,
        src: Tensor,
        dst: Tensor,
        extra: Iterable[Tensor] = (),
        query_time: Tensor | None = None,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor] | None]:
        query_parts = [src, dst, *extra]
        query = torch.cat(query_parts).unique()
        n_id, edge_index, event_id = self.neighbor_loader(query)
        self.association[n_id] = torch.arange(n_id.size(0), device=self.device_ref)
        memory, last_update = self.memory(n_id)
        if event_id.numel():
            edge_time = self.history_t[event_id]
            edge_msg = self.history_msg[event_id]
        else:
            edge_time = torch.empty(0, dtype=torch.long, device=self.device_ref)
            edge_msg = torch.empty(
                (0, self.raw_msg_dim), dtype=torch.float, device=self.device_ref
            )
        result = self.embedding(
            memory,
            last_update,
            edge_index,
            edge_time,
            edge_msg,
            query_time=query_time,
            return_attention=return_attention,
        )
        attention = None
        if return_attention:
            embedding, (att_edge_index, weights) = result
            attention = {
                "edge_index": att_edge_index,
                "weights": weights,
                "event_id": event_id,
                "node_ids": n_id,
            }
        else:
            embedding = result
        return embedding, self.association, attention

    def logits(
        self,
        src: Tensor,
        dst: Tensor,
        neg_dst: Tensor | None = None,
        query_time: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        extras = () if neg_dst is None else (neg_dst,)
        embedding, association, _ = self._query_embeddings(
            src, dst, extras, query_time=query_time
        )
        positive = self.scorer(embedding[association[src]], embedding[association[dst]])
        negative = None
        if neg_dst is not None:
            negative = self.scorer(
                embedding[association[src]], embedding[association[neg_dst]]
            )
        return positive, negative

    def observe(self, batch: Batch) -> None:
        """Update memory and recent-neighbor state after prediction."""
        self.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        stop = self._history_length + batch.size
        self._ensure_history_capacity(stop)
        self._history_t[self._history_length : stop].copy_(batch.t.detach())
        self._history_msg[self._history_length : stop].copy_(batch.msg.detach())
        self._history_length = stop
        self.neighbor_loader.insert(batch.src, batch.dst)

    @torch.no_grad()
    def candidate_transition(
        self, batch: Batch, node_ids: Tensor
    ) -> tuple["TGNDetector", Tensor, Tensor]:
        """Return the exact post-batch shadow state without committing it.

        A state-integrity monitor can inspect the transition proposed by the
        current batch and then either commit the returned detector or reject
        that same transition. The live detector remains unchanged.
        """
        node_ids = node_ids.to(self.device_ref).unique()
        before, _ = self.memory(node_ids)
        candidate = self.fork()
        candidate.observe(batch)
        after, _ = candidate.memory(node_ids)
        return candidate, before.detach().clone(), after.detach().clone()

    def positive_anomaly(
        self, src: Tensor, dst: Tensor, query_time: Tensor | None = None
    ) -> Tensor:
        positive, _ = self.logits(src, dst, query_time=query_time)
        return F.softplus(-positive)

    def edge_type_logits(self, batch: Batch) -> Tensor:
        if self.edge_decoder is None:
            raise RuntimeError("This model has no edge-type reconstruction decoder")
        _, _, type_logits = self.reconstruction_outputs(batch)
        assert type_logits is not None
        return type_logits

    def reconstruction_outputs(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        extras = () if batch.neg_dst is None else (batch.neg_dst,)
        embedding, association, _ = self._query_embeddings(
            batch.src, batch.dst, extras, query_time=batch.t
        )
        source = embedding[association[batch.src]]
        destination = embedding[association[batch.dst]]
        positive = self.scorer(source, destination)
        negative = None
        if batch.neg_dst is not None:
            negative = self.scorer(
                source, embedding[association[batch.neg_dst]]
            )
        type_logits = (
            None
            if self.edge_decoder is None
            else self.edge_decoder(source, destination)
        )
        return positive, negative, type_logits

    def anomaly(self, batch: Batch) -> Tensor:
        """Pre-update multi-task reconstruction error for observed events."""
        positive, _, type_logits = self.reconstruction_outputs(batch)
        link_error = F.softplus(-positive)
        if self.edge_type_loss_weight <= 0:
            return self.link_loss_weight * link_error
        if batch.edge_type is None:
            raise ValueError("Edge-type labels are required for reconstruction scoring")
        assert type_logits is not None
        type_error = F.cross_entropy(type_logits, batch.edge_type, reduction="none")
        return (
            self.edge_type_loss_weight * type_error
            + self.link_loss_weight * link_error
        )

    def training_loss(self, batch: Batch) -> Tensor:
        positive, negative, type_logits = self.reconstruction_outputs(batch)
        link_loss = positive.new_zeros(())
        if self.link_loss_weight > 0:
            if negative is None:
                raise ValueError(
                    "Negative destinations are required when link_loss_weight is positive"
                )
            link_loss = F.softplus(-positive).mean() + F.softplus(negative).mean()
        if self.edge_type_loss_weight <= 0:
            return self.link_loss_weight * link_loss
        if batch.edge_type is None:
            raise ValueError("Training requires edge-type labels")
        assert type_logits is not None
        type_loss = F.cross_entropy(type_logits, batch.edge_type)
        return (
            self.edge_type_loss_weight * type_loss
            + self.link_loss_weight * link_loss
        )

    def attention_trace(
        self, src: Tensor, dst: Tensor, query_time: Tensor
    ) -> dict[str, Tensor]:
        """Return neighbor-event attention for diagnostic attribution."""
        _, _, attention = self._query_embeddings(
            src, dst, query_time=query_time, return_attention=True
        )
        assert attention is not None
        return attention

    def temporal_state_dict(self) -> dict[str, Any]:
        def export_store(store: dict) -> dict[int, tuple[Tensor, ...]]:
            return {
                int(node): tuple(value.detach().cpu().clone() for value in values)
                for node, values in store.items()
            }

        return {
            "training": bool(self.training),
            "msg_s_store": export_store(self.memory.msg_s_store),
            "msg_d_store": export_store(self.memory.msg_d_store),
            "neighbors": self.neighbor_loader.neighbors.detach().cpu().clone(),
            "edge_ids": self.neighbor_loader.e_id.detach().cpu().clone(),
            "loader_assoc": self.neighbor_loader._assoc.detach().cpu().clone(),
            "current_edge_id": int(self.neighbor_loader.cur_e_id),
            "history_t": self.history_t.detach().cpu().clone(),
            "history_msg": self.history_msg.detach().cpu().clone(),
        }

    def load_temporal_state_dict(self, state: dict[str, Any]) -> None:
        device = self.device_ref
        # Restore the saved mode without invoking ``TGNMemory.train``.  Its
        # train->eval hook flushes pending messages; that is correct during an
        # ordinary run but would mutate parameters/buffers already restored
        # from the checkpoint.  Directly restoring the standard ``training``
        # flags makes this deserialisation operation side-effect free.
        restored_training = bool(state.get("training", False))
        for module in self.modules():
            module.training = restored_training

        def import_store(store: dict) -> dict[int, tuple[Tensor, ...]]:
            return {
                int(node): tuple(value.to(device).clone() for value in values)
                for node, values in store.items()
            }

        self.memory.msg_s_store = import_store(state["msg_s_store"])
        self.memory.msg_d_store = import_store(state["msg_d_store"])
        self.neighbor_loader.neighbors.copy_(state["neighbors"].to(device))
        self.neighbor_loader.e_id.copy_(state["edge_ids"].to(device))
        self.neighbor_loader._assoc.copy_(state["loader_assoc"].to(device))
        self.neighbor_loader.cur_e_id = int(state["current_edge_id"])
        self._set_history(
            state["history_t"].to(device).clone(),
            state["history_msg"].to(device).clone(),
        )


def iter_batches(
    src: Tensor,
    dst: Tensor,
    t: Tensor,
    msg: Tensor,
    batch_size: int,
    neg_dst: Tensor | None = None,
    edge_type: Tensor | None = None,
) -> Iterable[Batch]:
    for start in range(0, src.numel(), batch_size):
        stop = min(start + batch_size, src.numel())
        yield Batch(
            src=src[start:stop],
            dst=dst[start:stop],
            t=t[start:stop],
            msg=msg[start:stop],
            neg_dst=None if neg_dst is None else neg_dst[start:stop],
            edge_type=None if edge_type is None else edge_type[start:stop],
        )
