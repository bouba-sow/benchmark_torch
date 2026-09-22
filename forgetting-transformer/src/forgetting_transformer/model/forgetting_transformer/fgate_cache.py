from typing import List, Tuple, Optional, Any, Dict
import torch

class FgateDynamicCache:
    """
    A cache that grows dynamically as more tokens are generated.
    Custom cache for Forgetting Transformer that does not inherit from transformers.Cache.
    """

    def __init__(self, num_hidden_layers: Optional[int] = None) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.log_fgate_cache: List[torch.Tensor] = []
        self.key_shift_cache: List[torch.Tensor] = []
        self.value_shift_cache: List[torch.Tensor] = []
        self._seen_tokens = 0

    def update_shift_cache(
        self,
        key_shift_state: torch.Tensor,
        value_shift_state: torch.Tensor,
        layer_idx,
    ):
        assert layer_idx == len(self.key_shift_cache) == len(self.value_shift_cache)
        self.key_shift_cache.append(key_shift_state)
        self.value_shift_cache.append(value_shift_state)

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (self.key_cache[layer_idx], self.value_cache[layer_idx], self.log_fgate_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx], self.log_fgate_cache[layer_idx])

    def __len__(self):
        return len(self.key_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        log_fgate_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert log_fgate_states.ndim == 3, f"log_fgate must be (B, H, T), but get {log_fgate_states.size()}"
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            self.log_fgate_cache.append(log_fgate_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            self.log_fgate_cache[layer_idx] = torch.cat([self.log_fgate_cache[layer_idx], log_fgate_states], dim=-1)

        return self.key_cache[layer_idx], self.value_cache[layer_idx], self.log_fgate_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        return None

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], ...]:
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx], self.value_cache[layer_idx], self.log_fgate_cache[layer_idx]),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None, num_layers: Optional[int] = None) -> "FgateDynamicCache":
        """
        Converts a cache in the legacy cache format into an equivalent FgateDynamicCache.
        
        Args:
            past_key_values: Optional legacy cache format
            num_layers: Not used in this implementation
        
        Returns:
            FgateDynamicCache instance
        """
        cache = cls()
        
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states, log_fgate_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, log_fgate_states, layer_idx)
        
        return cache

    def crop(self, max_length: int):
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self._seen_tokens = max_length
        for idx in range(len(self.key_cache)):
            self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
            self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]
            self.log_fgate_cache[idx] = self.log_fgate_cache[idx][..., :max_length]

    def batch_split(self, full_batch_size: int, split_size: int) -> List["FgateDynamicCache"]:
        out = []
        for i in range(0, full_batch_size, split_size):
            current_split = FgateDynamicCache()
            current_split._seen_tokens = self._seen_tokens
            current_split.key_cache = [tensor[i : i + split_size] for tensor in self.key_cache]
            current_split.value_cache = [tensor[i : i + split_size] for tensor in self.value_cache]
            current_split.log_fgate_cache = [tensor[i : i + split_size] for tensor in self.log_fgate_cache]
            out.append(current_split)
        return out

    @classmethod
    def from_batch_splits(cls, splits: List["FgateDynamicCache"]) -> "FgateDynamicCache":
        cache = cls()
        for idx in range(len(splits[0])):
            layer_keys = torch.cat([current.key_cache[idx] for current in splits], dim=0)
            layer_values = torch.cat([current.value_cache[idx] for current in splits], dim=0)
            layer_log_fgates = torch.cat([current.log_fgate_cache[idx] for current in splits], dim=0)
            cache.update(layer_keys, layer_values, layer_log_fgates, idx)
        return cache

    def batch_repeat_interleave(self, repeats: int):
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.value_cache[layer_idx] = self.value_cache[layer_idx].repeat_interleave(repeats, dim=0)
            self.log_fgate_cache[layer_idx] = self.log_fgate_cache[layer_idx].repeat_interleave(repeats, dim=0)

    def batch_select_indices(self, indices: torch.Tensor):
        for layer_idx in range(len(self)):
            self.key_cache[layer_idx] = self.key_cache[layer_idx][indices, ...]
            self.value_cache[layer_idx] = self.value_cache[layer_idx][indices, ...]
            self.log_fgate_cache[layer_idx] = self.log_fgate_cache[layer_idx][indices, ...]