from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from qwen_mm_token_probe.hf_qwen import (
    collate_prompt_inputs,
    generate_batch_from_prompts,
    generate_from_prompt,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    all_special_ids: ClassVar[list[int]] = [0, 2, 3, 99]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return " ".join(str(token_id) for token_id in token_ids if token_id not in self.all_special_ids)


class FakeGenerateModel(torch.nn.Module):
    def __init__(self, generated_tail: list[list[int]], *, eos_token_id: int | list[int] = 2) -> None:
        super().__init__()
        self.generation_config = SimpleNamespace(eos_token_id=eos_token_id, pad_token_id=0)
        self.generated_tail = generated_tail
        self.generate_calls = 0
        self.seen_inputs: dict[str, torch.Tensor] = {}
        self.seen_kwargs: dict[str, object] = {}

    def generate(self, **kwargs: object) -> torch.Tensor:
        self.generate_calls += 1
        input_ids = kwargs["input_ids"]
        assert torch.is_tensor(input_ids)
        self.seen_inputs = {
            key: value.clone()
            for key, value in kwargs.items()
            if torch.is_tensor(value)
        }
        self.seen_kwargs = {key: value for key, value in kwargs.items() if not torch.is_tensor(value)}
        tail_width = max(len(row) for row in self.generated_tail)
        tails = [row + [0] * (tail_width - len(row)) for row in self.generated_tail]
        tail_tensor = torch.tensor(tails, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, tail_tensor], dim=1)


def test_collate_left_pads_sequences_and_type_ids_without_mutating_inputs() -> None:
    first = {
        "input_ids": torch.tensor([[11, 12]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "mm_token_type_ids": torch.tensor([[3, 4]], dtype=torch.long),
        "token_type_ids": torch.tensor([[5, 6]], dtype=torch.long),
    }
    second = {
        "input_ids": torch.tensor([[21, 22, 23]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        "mm_token_type_ids": torch.tensor([[7, 8, 9]], dtype=torch.long),
        "token_type_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
    }
    first_input_ids = first["input_ids"].clone()
    first_mm_types = first["mm_token_type_ids"].clone()
    second_input_ids = second["input_ids"].clone()

    batch = collate_prompt_inputs([first, second], pad_token_id=0)

    assert batch["input_ids"].tolist() == [[0, 11, 12], [21, 22, 23]]
    assert batch["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]
    assert batch["mm_token_type_ids"].tolist() == [[0, 3, 4], [7, 8, 9]]
    assert batch["token_type_ids"].tolist() == [[0, 5, 6], [1, 2, 3]]
    assert first["input_ids"].equal(first_input_ids)
    assert first["mm_token_type_ids"].equal(first_mm_types)
    assert second["input_ids"].equal(second_input_ids)


def test_collate_concatenates_multi_image_tensors_in_sample_order() -> None:
    first = {
        "input_ids": torch.tensor([[1, 2]]),
        "pixel_values": torch.tensor([[10.0], [11.0]]),
        "image_grid_thw": torch.tensor([[1, 1, 2], [1, 2, 1]]),
    }
    second = {
        "input_ids": torch.tensor([[3]]),
        "pixel_values": torch.tensor([[20.0], [21.0], [22.0]]),
        "image_grid_thw": torch.tensor([[2, 1, 2]]),
    }

    batch = collate_prompt_inputs([first, second], pad_token_id=0)

    assert batch["pixel_values"].flatten().tolist() == [10.0, 11.0, 20.0, 21.0, 22.0]
    assert batch["image_grid_thw"].tolist() == [[1, 1, 2], [1, 2, 1], [2, 1, 2]]


def test_generate_batch_calls_model_once_and_strips_only_eos_and_pad_tails() -> None:
    first = {
        "input_ids": torch.tensor([[10, 11]]),
        "attention_mask": torch.tensor([[1, 1]]),
        "pixel_values": torch.tensor([[1.0]]),
        "image_grid_thw": torch.tensor([[1, 1, 1]]),
    }
    second = {
        "input_ids": torch.tensor([[20, 21, 22]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "pixel_values": torch.tensor([[2.0], [3.0]]),
        "image_grid_thw": torch.tensor([[1, 1, 2]]),
    }
    original_first_ids = first["input_ids"].clone()
    original_second_pixels = second["pixel_values"].clone()
    model = FakeGenerateModel([[5, 2, 0], [6, 99, 0]])

    results = generate_batch_from_prompts(
        model=model,
        tokenizer=FakeTokenizer(),
        prompt_inputs=[first, second],
        max_new_tokens=128,
    )

    assert results == [([5], "5"), ([6], "6")]
    assert model.generate_calls == 1
    assert model.seen_kwargs == {"max_new_tokens": 128, "do_sample": False}
    assert model.seen_inputs["input_ids"].tolist() == [[0, 10, 11], [20, 21, 22]]
    assert model.seen_inputs["attention_mask"].tolist() == [[0, 1, 1], [1, 1, 1]]
    assert model.seen_inputs["pixel_values"].flatten().tolist() == [1.0, 2.0, 3.0]
    assert first["input_ids"].equal(original_first_ids)
    assert second["pixel_values"].equal(original_second_pixels)


def test_batch_size_one_matches_single_generation_tail_semantics_with_eos_list() -> None:
    prompt = {
        "input_ids": torch.tensor([[10, 11]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    tokenizer = FakeTokenizer()
    batch_model = FakeGenerateModel([[5, 99, 3, 0]], eos_token_id=[2, 3])
    single_model = FakeGenerateModel([[5, 99, 3, 0]], eos_token_id=[2, 3])

    batch_result = generate_batch_from_prompts(
        model=batch_model,
        tokenizer=tokenizer,
        prompt_inputs=[prompt],
        max_new_tokens=16,
    )[0]
    single_result = generate_from_prompt(
        model=single_model,
        tokenizer=tokenizer,
        prompt_inputs=prompt,
        max_new_tokens=16,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    assert batch_result == single_result == ([5], "5")
    assert batch_model.generate_calls == single_model.generate_calls == 1


def test_model_eos_configuration_takes_precedence_over_tokenizer_eos() -> None:
    prompt = {
        "input_ids": torch.tensor([[10, 11]]),
        "attention_mask": torch.tensor([[1, 1]]),
    }
    tokenizer = FakeTokenizer()
    tokenizer.eos_token_id = 2
    batch_model = FakeGenerateModel([[5, 2, 6, 3, 0]], eos_token_id=3)
    single_model = FakeGenerateModel([[5, 2, 6, 3, 0]], eos_token_id=3)

    batch_result = generate_batch_from_prompts(
        model=batch_model,
        tokenizer=tokenizer,
        prompt_inputs=[prompt],
        max_new_tokens=16,
    )[0]
    single_result = generate_from_prompt(
        model=single_model,
        tokenizer=tokenizer,
        prompt_inputs=prompt,
        max_new_tokens=16,
        do_sample=False,
        temperature=None,
        top_p=None,
    )

    assert batch_result == single_result == ([5, 2, 6], "5 6")


def test_tiny_random_qwen35_batch_matches_single_for_left_padded_image_prompts() -> None:
    transformers = pytest.importorskip("transformers")
    config_class = getattr(transformers, "Qwen3_5Config", None)
    model_class = getattr(transformers, "Qwen3_5ForConditionalGeneration", None)
    if config_class is None or model_class is None:
        pytest.skip("installed transformers does not include Qwen3.5")
    try:
        from transformers.models.qwen3_5.configuration_qwen3_5 import (
            Qwen3_5TextConfig,
            Qwen3_5VisionConfig,
        )
    except ImportError:
        pytest.skip("installed transformers does not include Qwen3.5 config classes")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1847)
            text_config = Qwen3_5TextConfig(
                vocab_size=64,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                max_position_embeddings=64,
                layer_types=["full_attention", "full_attention"],
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=[62, 63],
            )
            vision_config = Qwen3_5VisionConfig(
                depth=1,
                hidden_size=8,
                intermediate_size=16,
                num_heads=2,
                in_channels=3,
                patch_size=2,
                spatial_merge_size=1,
                temporal_patch_size=1,
                out_hidden_size=16,
                num_position_embeddings=16,
            )
            model_config = config_class(
                text_config=text_config,
                vision_config=vision_config,
                image_token_id=10,
                video_token_id=11,
                vision_start_token_id=12,
                vision_end_token_id=13,
            )
            batch_model = model_class(model_config).eval()
            single_model = model_class(model_config).eval()
            single_model.load_state_dict(batch_model.state_dict())
            for model in (batch_model, single_model):
                model.generation_config.pad_token_id = 0
                model.generation_config.eos_token_id = [62, 63]

            prompts = [
                {
                    "input_ids": torch.tensor([[5, 12, 10, 13, 6]]),
                    "attention_mask": torch.ones((1, 5), dtype=torch.long),
                    "mm_token_type_ids": torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.int),
                    "pixel_values": torch.randn((1, 12)),
                    "image_grid_thw": torch.tensor([[1, 1, 1]]),
                },
                {
                    "input_ids": torch.tensor([[5, 12, 10, 10, 13, 6, 7]]),
                    "attention_mask": torch.ones((1, 7), dtype=torch.long),
                    "mm_token_type_ids": torch.tensor([[0, 0, 1, 1, 0, 0, 0]], dtype=torch.int),
                    "pixel_values": torch.randn((2, 12)),
                    "image_grid_thw": torch.tensor([[1, 1, 2]]),
                },
            ]
            tokenizer = FakeTokenizer()
            tokenizer.eos_token_id = 62
            tokenizer.all_special_ids = [0, 62, 63]

            batched_results = generate_batch_from_prompts(
                model=batch_model,
                tokenizer=tokenizer,
                prompt_inputs=prompts,
                max_new_tokens=4,
            )
            single_results = [
                generate_from_prompt(
                    model=single_model,
                    tokenizer=tokenizer,
                    prompt_inputs=prompt,
                    max_new_tokens=4,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                )
                for prompt in prompts
            ]

            assert batched_results == single_results
    finally:
        torch.set_num_threads(previous_threads)


def test_collate_rejects_unknown_processor_input_keys() -> None:
    with pytest.raises(ValueError, match="unsupported prepared prompt input key.*video_grid_thw"):
        collate_prompt_inputs(
            [{"input_ids": torch.tensor([[1]]), "video_grid_thw": torch.tensor([[1, 1, 1]])}],
            pad_token_id=0,
        )
